#!/usr/bin/bash
SIGULARITY_IMAGES_DIR=${SCRATCH}/singularity_images
DOCKERHUB_USER=exc1ted
REPO_NAME=exc1ted
IMAGE_TAG=pred-gen1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "ls ${SIGULARITY_IMAGES_DIR}"

echo "rm ${SIGULARITY_IMAGES_DIR}/${REPO_NAME}_${IMAGE_TAG}.sif"

echo "singularity pull --dir ${SIGULARITY_IMAGES_DIR} docker://${DOCKERHUB_USER}/${REPO_NAME}:${IMAGE_TAG}"

# srun -p amdv100 -N 1 --pty bash
srun -p intelv100,amdv100 --nodes 1 --job-name interactive \
  --time 04:00:00 --ntasks 1 --cpus-per-task 32 --mem 64G --pty bash