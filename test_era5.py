import os
import time
import h5py
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
from scipy.interpolate import griddata
from tqdm import tqdm

import io
import torch

from src.utils.common import set_torch_env, get_state_dict
from src.layers.cuda_inference import replicate_pad
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
  read_sps_remaining, read_ip_remaining, write_ip

def convert_nc_to_hdf5(nc_file, hdf5_file):
  """
  Convert a NetCDF (.nc) file to HDF5 (.h5) format.

  Parameters:
    nc_file (str): Path to the input NetCDF file.
    hdf5_file (str): Path to the output HDF5 file.
  """
  # Open the NetCDF file
  dataset = xr.open_dataset(nc_file)

  # Create an HDF5 file
  with h5py.File(hdf5_file, 'w') as hdf5_f:
    for var_name, da in dataset.data_vars.items():
      data = da.values[0:] # Convert xarray DataArray to NumPy array
      hdf5_f.create_dataset(var_name, data=data)

def spatial_interpolation(data, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, start_idx=0, end_idx=None):
  data = data[start_idx:end_idx]
  num_time_steps = data.shape[0]
  data_interpolated = np.empty((num_time_steps, lon_target_grid.shape[0], lon_target_grid.shape[1]))
  points = np.column_stack((lat_source_grid.ravel(), lon_source_grid.ravel()))
  for t_idx in range(num_time_steps):
    values = data[t_idx].ravel()
    data_interpolated[t_idx] = griddata(points, values, (lat_target_grid, lon_target_grid), method='linear')
  return data_interpolated

def interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_file, output_file):
  # Load reanalysis and ensemble datasets
  ds_reanalysis = xr.open_dataset(reanalysis_file)
  ds_ensemble = xr.open_dataset(ensemble_file)

  # Extract coordinates from reanalysis dataset (target grid)
  time_target = ds_reanalysis['valid_time'].values
  lat_target = ds_reanalysis['latitude'].values
  lon_target = ds_reanalysis['longitude'].values

  # Extract coordinates and spread variable from ensemble dataset (source grid)
  time_source = ds_ensemble['valid_time'].values
  lat_source = ds_ensemble['latitude'].values
  lon_source = ds_ensemble['longitude'].values

  # shape: (Time, Latitude, Longitude)
  assert len(list(ds_ensemble.data_vars)) == 1
  for var_name, da in ds_ensemble.data_vars.items():
    data_source = da.values  # Convert xarray DataArray to NumPy array
    # Step 1: Interpolate Spatial Dims
    # handle longitude wrap-up at 360
    # Src
    lon_source_extended = np.concatenate((lon_source, lon_source[0:1] + 360), axis=0)
    lat_source_grid, lon_source_grid = np.meshgrid(lat_source, lon_source_extended, indexing='ij')
    data_extended = np.concatenate((data_source, data_source[:, :, 0:1]), axis=2)
    # Dst
    lat_target_grid, lon_target_grid = np.meshgrid(lat_target, lon_target, indexing='ij')

    num_time_steps = 8
    num_jobs = (data_extended.shape[0] + num_time_steps - 1) // num_time_steps

    # mp
    with mp.Pool(processes=32) as pool:  # Adjust processes as needed
      results = [pool.apply_async(spatial_interpolation,
        (data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0]))
      ) for idx in range(num_jobs)]
      results = [result.get() for result in results]
    
    # for loop
    # results = [spatial_interpolation(
    #   data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0])
    # ) for idx in range(num_jobs)]

    results = np.concatenate(results, axis=0)

    # Step 2: Interpolate Temporal Dim
    ds_interp_space = xr.Dataset(
      {
        var_name: (['valid_time', 'latitude', 'longitude'], results)
      },
      coords={
        'valid_time': time_source,
        'latitude': lat_target,
        'longitude': lon_target
      }
    )
    # Interpolate in time to match reanalysis time grid
    ds_interp_time = ds_interp_space.interp(
      valid_time=time_target, 
      method="linear")
    ds_output = ds_interp_time.ffill(dim="valid_time")

    # Save to new hdf5 file
    with h5py.File(output_file, 'w') as hdf5_f:
      for var_name, da in ds_output.data_vars.items():
        data = da.values.astype(np.float32)[0:] # Convert xarray DataArray to NumPy array
        hdf5_f.create_dataset(var_name, data=data)

class DcvcCompressor:
  def __init__(self):
    set_torch_env()
    self.device = 'cuda:0'
    self.force_zero_thres = 0.12

    self.model_path_i = '/capstor/scratch/cscs/ljiayong/workspace/DCVC/pretrained/cvpr2025_image.pth.tar'
    self.model_path_p = '/capstor/scratch/cscs/ljiayong/workspace/DCVC/pretrained/cvpr2025_video.pth.tar'

    self.i_frame_net = DMCI()
    i_state_dict = get_state_dict(self.model_path_i)
    self.i_frame_net.load_state_dict(i_state_dict)
    self.i_frame_net = self.i_frame_net.to(self.device)
    self.i_frame_net.eval()
    self.i_frame_net.update(self.force_zero_thres)
    self.i_frame_net.half()

    self.p_frame_net = DMC()
    p_state_dict = get_state_dict(self.model_path_p)
    self.p_frame_net.load_state_dict(p_state_dict)
    self.p_frame_net = self.p_frame_net.to(self.device)
    self.p_frame_net.eval()
    self.p_frame_net.update(self.force_zero_thres)
    self.p_frame_net.half()
  
  def compress(self, data, qp, output_path):
    '''
    NN
    '''
    # setting
    qp_i = qp
    qp_p = qp_i
    reset_interval = 64
    intra_period = -1
    frame_num, height, width = data.shape # (744, 721, 1440)
    padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
    use_two_entropy_coders = height * width > 1280 * 720
    self.i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    self.p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    output_buff = io.BytesIO()
    sps_helper = SPSHelper()

    self.p_frame_net.set_curr_poc(0)
    with torch.no_grad():
      last_qp = 0
      for frame_idx in range(frame_num):
        data_tensor = torch.Tensor(data[frame_idx:frame_idx+1]).half().to(self.device)
        data_tensor = data_tensor.unsqueeze(1).repeat(1,3,1,1) # [N, C, H, W] with FP32 in Range [0, 1]

        torch.cuda.synchronize(device=self.device)

        # pad if necessary
        data_tensor = replicate_pad(data_tensor, padding_b, padding_r)

        is_i_frame = False
        if frame_idx == 0 or (intra_period > 0 and frame_idx % intra_period == 0):
          is_i_frame = True
          curr_qp = qp_i
          sps = {
              'sps_id': -1,
              'height': height,
              'width': width,
              'ec_part': 1 if use_two_entropy_coders else 0,
              'use_ada_i': 0,
          }
          encoded = self.i_frame_net.compress(data_tensor, qp_i)
          self.p_frame_net.clear_dpb()
          self.p_frame_net.add_ref_frame(None, encoded['x_hat'])
        else:
          fa_idx = index_map[frame_idx % 8]
          if reset_interval > 0 and frame_idx % reset_interval == 1:
            use_ada_i = 1
            self.p_frame_net.prepare_feature_adaptor_i(last_qp)
          else:
            use_ada_i = 0
          curr_qp = self.p_frame_net.shift_qp(qp_p, fa_idx)
          sps = {
              'sps_id': -1,
              'height': height,
              'width': width,
              'ec_part': 1 if use_two_entropy_coders else 0,
              'use_ada_i': use_ada_i,
          }

          encoded = self.p_frame_net.compress(data_tensor, curr_qp)
          last_qp = curr_qp

        sps_id, sps_new = sps_helper.get_sps_id(sps)
        sps['sps_id'] = sps_id
        sps_bytes = 0
        if sps_new:
          sps_bytes = write_sps(output_buff, sps)
        stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])

        torch.cuda.synchronize(device=self.device)

    with open(output_path, "wb") as output_file:
      bytes_buffer = output_buff.getbuffer()
      output_file.write(bytes_buffer)
      total_bytes = bytes_buffer.nbytes
      bytes_buffer.release()
    output_buff.close()
    return total_bytes
  
  def decompress(self, compressed_file, frame_num, height, width):
    sps_helper = SPSHelper()
    with open(compressed_file, "rb") as input_file:
      input_buff = io.BytesIO(input_file.read())
    decoded_frame_number = 0

    self.p_frame_net.set_curr_poc(0)
    with torch.no_grad():
      data_hat = torch.zeros(frame_num, height, width, dtype = torch.float)
      while decoded_frame_number < frame_num:
        torch.cuda.synchronize(device=self.device)

        header = read_header(input_buff)
        while header['nal_type'] == NalType.NAL_SPS:
          sps = read_sps_remaining(input_buff, header['sps_id'])
          sps_helper.add_sps_by_id(sps)
          header = read_header(input_buff)
          continue
        sps_id = header['sps_id']

        sps = sps_helper.get_sps_by_id(sps_id)
        qp, bit_stream = read_ip_remaining(input_buff)

        if header['nal_type'] == NalType.NAL_I:
          decoded = self.i_frame_net.decompress(bit_stream, sps, qp)
          self.p_frame_net.clear_dpb()
          self.p_frame_net.add_ref_frame(None, decoded['x_hat'])
        elif header['nal_type'] == NalType.NAL_P:
          if sps['use_ada_i']:
            self.p_frame_net.reset_ref_feature()
          decoded = self.p_frame_net.decompress(bit_stream, sps, qp)

        recon_frame = decoded['x_hat']

        x_hat = recon_frame[:, :, :height, :width].float()
        data_hat[decoded_frame_number:decoded_frame_number+1] = x_hat.mean(dim = 1)

        torch.cuda.synchronize(device=self.device)
        decoded_frame_number += 1
    input_buff.close()
    return data_hat
  
  def run_benchmark(self, data, error_bound, compressed_file_path):
    data, error_bound = data[0:5], error_bound[0:5]
    print(f'[INFO] Starting Data Compression......')
    qp = 60
    frames, height, width = data.shape # (744, 721, 1440)

    '''
    preprocess
    '''# if clip_extreme:
    #   quantile = 0.2
    #   range_factor = 2
    #   Q1 = np.quantile(data, quantile)
    #   Q3 = np.quantile(data, 1-quantile)
    #   IQR = Q3 - Q1
    #   lower_bound = Q1 - range_factor * IQR
    #   upper_bound = Q3 + range_factor * IQR

    #   data_extreme_mask = (data < lower_bound) | (data > upper_bound)
    #   data_extreme_ratio = data_extreme_mask.sum()/data_extreme_mask.size
    #   data_extreme_positions = np.flatnonzero(data_extreme_mask)
    #   data_extreme_values = data[data_extreme_mask]

    #   data_clipped = np.clip(data, lower_bound, upper_bound)
    # else:
    #   data_extreme_positions = np.array([], dtype=np.int64)
    #   data_extreme_values = np.array([], dtype=np.float32)

    #   data_clipped = data


    # Step 2: Scale the input
    min_val = data.min()
    max_val = data.max()

    if max_val == min_val:
      data_scaled = data - min_val
    else:
      data_scaled = (data - min_val) / (max_val - min_val)


    total_bytes = self.compress(data_scaled, qp, compressed_file_path)
    data_hat_scaled = self.decompress(compressed_file_path, frames, height, width)

    if max_val == min_val:
      data_hat = data_hat_scaled + min_val
    else:
      data_hat = data_hat_scaled * (max_val - min_val) + min_val
    import pdb;pdb.set_trace()


def compress_hdf5_dcvc_pointwise(input_hdf5, input_uncertainty_hdf5, compressed_file_path, ebcc_pointwise_max_error_ratio):
  # compression and compression time
  compression_start_time = time.time()
  
  with h5py.File(input_hdf5, 'r') as hdf5_in:
    with h5py.File(input_uncertainty_hdf5, 'r') as hdf5_uncertainty_in:
      assert len(list(hdf5_in.keys())) == 1
      var_name = list(hdf5_in.keys())[0]
      data = np.array(hdf5_in[var_name])  # Read dataset, 1 month data: (744, 721, 1440)
      error_bound = np.array(hdf5_uncertainty_in[var_name]) * ebcc_pointwise_max_error_ratio
      compressor = DcvcCompressor()
      best_compressed_bytes = compressor.run_benchmark(data, error_bound, compressed_file_path)

  compression_end_time = time.time()
  compression_time = compression_end_time - compression_start_time

  input_size = os.path.getsize(input_hdf5)
  compression_ratio = input_size/best_compressed_bytes
  compression_bandwidth = input_size/1e6/compression_time
  import pdb;pdb.set_trace()
  return compression_time, compression_ratio, compression_bandwidth

def run_dcvc_pointwise(output_path, variable, ebcc_pointwise_max_error_ratio):
  input_hdf5_file_path = os.path.join(output_path, f'{variable}.hdf5')
  input_uncertainty_file_path = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
  output_file_path = os.path.join(output_path, f'{variable}_compressed_dcvc_pointwise_ratio_{ebcc_pointwise_max_error_ratio}.bin')
  compression_time, compression_ratio, compression_bandwidth = compress_hdf5_dcvc_pointwise(input_hdf5_file_path, input_uncertainty_file_path, output_file_path, ebcc_pointwise_max_error_ratio)
  results = {
    'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
    'compression_ratio' : compression_ratio,
    'compression_time' : compression_time,
    'compression_bandwidth': compression_bandwidth,
  }
  import pdb;pdb.set_trace()
  return results

if __name__ == "__main__":
  variable_lst = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation"
  ]

  # global value
  for variable_idx in range(len(variable_lst)):
    variable = variable_lst[variable_idx]
    output_path = f'/capstor/scratch/cscs/ljiayong/workspace/DCVC/test'
    os.makedirs(output_path, exist_ok = True)

    # '''
    # Step 1: NetCDF to HDF5 without compression
    # '''
    # print(f'[INFO] Converting NetCDF to HDF5 for Variable {variable} ......')
    # nc_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    # hdf5_file = os.path.join(output_path, f'{variable}.hdf5')
    # convert_nc_to_hdf5(nc_file, hdf5_file)


    # '''
    # Step 2: Interpolate Ensemble Spread
    # '''
    # print(f'[INFO] Interpolating Ensemble Spread for Variable {variable} ......')
    # reanalysis_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    # ensemble_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/ensemble_spread/{variable}.nc'
    # output_file = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
    # interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_file, output_file)

    '''
    Param Combinations
    '''
    ebcc_pointwise_max_error_ratio = 1.
    
    '''
    Step 3: Run DCVC with Pointwise Error Bound
    '''
    params = (output_path, variable, ebcc_pointwise_max_error_ratio)
    run_dcvc_pointwise(*params)
  pass


