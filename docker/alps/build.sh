#!/bin/bash
IMAGE_NAME="dcvc"
IMAGE_TAG=latest

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# build
podman build -f Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} ${SCRIPT_DIR}/..

# enroot import
SQSH_DIR=/capstor/scratch/cscs/ljiayong/enroot_images
SQSH_FILE_NAME="${IMAGE_NAME//-/_}"
mkdir -p ${SQSH_DIR}
# remove sqsh if exists
rm -f ${SQSH_DIR}/${SQSH_FILE_NAME}.sqsh
enroot import -o ${SQSH_DIR}/${SQSH_FILE_NAME}.sqsh podman://${IMAGE_NAME}:${IMAGE_TAG}