#!/usr/bin/bash
SIGULARITY_IMAGES_DIR=${SCRATCH}/singularity_images
DOCKERHUB_USER=exc1ted
REPO_NAME=exc1ted
IMAGE_TAG=pred-gen1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

srun -p amda100,intelv100 --nodes 1 --job-name interactive --time 04:00:00 \
  --ntasks 1 --cpus-per-task 32 --mem 64G --gpus 2 \
  --pty singularity shell --shell /bin/bash --nv \
  --bind $SCRATCH:$SCRATCH \
  ${SIGULARITY_IMAGES_DIR}/${REPO_NAME}_${IMAGE_TAG}.sif