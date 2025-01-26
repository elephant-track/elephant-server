#! /usr/bin/env bash
set -e

USER_ID=${LOCAL_UID:-0}
GROUP_ID=${LOCAL_GID:-0}
NVIDIA_GID=${NVIDIA_GID:-0}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID, NVIDIA GID: ${NVIDIA_GID:-0}"
useradd -u $USER_ID -o -m user
if [ $GROUP_ID -ne 0 ]; then
  groupmod -o -g $GROUP_ID user
fi
if [ $NVIDIA_GID -ne 0 ]; then
  groupadd -g $NVIDIA_GID nvidia
  usermod -aG nvidia user
fi
export HOME=/home/user
export UWSGI_UID=user
export UWSGI_GID=user
export MKL_THREADING_LAYER=GNU
export GRPC_POLL_STRATEGY=epoll1

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
CONDA_PREFIX=$(conda info --base)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ -z "${AS_LOCAL_USER}" ]]; then
  "$@"
else
  exec /usr/sbin/gosu user "$@"
fi
