#!/bin/bash

app=$PWD

docker build -t istft-vocoder . && \
docker run -it --rm \
    --net=host --ipc=host \
    --gpus "all" \
    -v "$app":/app \
    istft-vocoder