# Build bundled liburing if needed
if [ ! -f "third_party/liburing/src/liburing.a" ]; then
    cd third_party/liburing
    ./configure --prefix=/usr/local
    make -j"$(nproc)"
    cd ../..
fi

# Detect CUDA
TORCH_CUDA=$(python -c "import torch; v=torch.version.cuda; print(v[:2] if v else '12')")
CUDA_HOME=""
for d in /usr/local/cuda-${TORCH_CUDA}.* /usr/local/cuda-${TORCH_CUDA} /usr/local/cuda; do
    if [ -x "$d/bin/nvcc" ]; then
        CUDA_HOME="$d"
        break
    fi
done

if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "Using CUDA_HOME=$CUDA_HOME"
else
    echo "WARNING: No system CUDA ${TORCH_CUDA}.x found, using default nvcc"
fi

# Detect GPU architecture
CUDA_ARCH_LIST=$(python -c "
import torch
if torch.cuda.is_available():
    print(';'.join(dict.fromkeys(
        f'{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}'
        for i in range(torch.cuda.device_count()
    )))
else:
    print('8.0')
")

# Build and install GriNNder
export LD_LIBRARY_PATH="$HOME/uGDS/build:$LD_LIBRARY_PATH"
GRINNDER_USE_UGDS=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CUDA_ARCH_LIST}}" \
    pip install -e . --no-build-isolation