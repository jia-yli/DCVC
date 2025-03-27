#!/bin/bash

IMAGE_NAME="exc1ted/exc1ted:pred-gen1"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ "$(docker image ls -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
  docker image rm ${IMAGE_NAME}
fi

docker build -t ${IMAGE_NAME} ${SCRIPT_DIR}
docker push ${IMAGE_NAME}