python test_video_single.py \
  --model_path_i /capstor/scratch/cscs/ljiayong/workspace/DCVC/pretrained/cvpr2025_image.pth.tar \
  --model_path_p /capstor/scratch/cscs/ljiayong/workspace/DCVC/pretrained/cvpr2025_video.pth.tar \
  --rate_num 2 \
  --test_config ./dataset_config_test.json \
  --cuda 1 \
  -w 1 \
  --write_stream 1 \
  --force_zero_thres 0.12 \
  --output_path ./results/output_test_single.json \
  --force_intra_period -1 \
  --reset_interval 64 \
  --force_frame_num -1 \
  --check_existing 0 \
  --verbose 0

# python test_video.py \
#   --model_path_i /capstor/scratch/cscs/ljiayong/workspace/DCVC/pretrained/cvpr2025_image.pth.tar \
#   --model_path_p /capstor/scratch/cscs/ljiayong/workspace/DCVC/pretrained/cvpr2025_video.pth.tar \
#   --rate_num 2 \
#   --test_config ./dataset_config_test.json \
#   --cuda 1 \
#   -w 4 \
#   --write_stream 1 \
#   --force_zero_thres 0.12 \
#   --output_path ./results/output_test.json \
#   --force_intra_period -1 \
#   --reset_interval 64 \
#   --force_frame_num -1 \
#   --check_existing 0 \
#   --verbose 0
