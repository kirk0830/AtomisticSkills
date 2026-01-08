# MatGL Agent Environment Setup (GPU)

This document describes the reproducible setup for the `matgl-agent` environment on the DGX Spark (Aarch64, NVIDIA GB10), enabling fully functional GPU acceleration for M3GNet, CHGNet, and TensorNet.

## 1. Environment Creation
Ensure you are in the `matgl-agent` environment:
```bash
conda activate matgl-agent
```

## 2. Install PyTorch 2.9.1+cu130
Install the specific nightly/test version compatible with CUDA 13.0 on Aarch64:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --upgrade
```
*Validated Version:* `2.9.1+cu130`

## 3. Build and Install DGL 2.5 (Source)
DGL must be compiled from source to support `sm_120` (GB10) and CUDA 13.0 on PyTorch 2.9.1.
**Option A: Use Helper Script** (Recommended)
```bash
chmod +x build_dgl_generic.sh
./build_dgl_generic.sh
```

**Option B: Manual Compilation**
If you need to replicate the build manually, follow these steps:

1. **Clone DGL Source**
```bash
1. **Locate DGL Source**
   The source is located at `~/projects/dgl_source`.
```bash
cd ~/projects/dgl_source
mkdir build && cd build
```

2. **Patch Source Code (Critical)**
   *File: `src/runtime/cuda/cuda_device_api.cc`* matches `src/runtime/c_runtime_api.cc` fix from previous steps.
   You must ensure the CUDA device API is forcefully linked.
   (If using the default main branch without patches, you might encounter `Device API cuda is not enabled`. See `build_dgl_generic.sh` or conversation history for patch details if needed).

3. **Configure CMake**
   Export necessary paths for CUDA 13.0 (on DGX Spark):
```bash
export MATH_LIBS="/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/math_libs/13.0/targets/sbsa-linux"
export CMAKE_PREFIX_PATH=$MATH_LIBS:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$MATH_LIBS/lib:$LD_LIBRARY_PATH
export CPATH=$MATH_LIBS/include:$CPATH
export CPLUS_INCLUDE_PATH=$MATH_LIBS/include:$CPLUS_INCLUDE_PATH
```
   Run CMake with manual architecture flags to avoid `compute_50` errors and target `sm_120`:
```bash
cmake -DUSE_CUDA=ON -DCMAKE_CXX_FLAGS="-fPIC" \
      -DCMAKE_PREFIX_PATH="$MATH_LIBS" \
      -DCUDA_cublas_LIBRARY="$MATH_LIBS/lib/libcublas.so" \
      -DCUDA_cusparse_LIBRARY="$MATH_LIBS/lib/libcusparse.so" \
      -DCUDA_CUDART_LIBRARY="/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0/lib64/libcudart.so" \
      -DTORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 12.0" \
      -DCUDA_ARCH_NAME=Manual \
      -DCUDA_ARCH_BIN="8.0 9.0 10.0 12.0" \
      -DCUDA_ARCH_PTX="12.0" \
      -DUSE_HUGECTR=OFF \
      -DBUILD_GRAPHBOLT=OFF \
      -DBUILD_SPARSE=OFF \
      ..
```

4. **Build and Install**
```bash
make -j8
# Install Python bindings
cd ../python
pip install .
# Overwrite libdgl.so with custom build
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp ../build/libdgl.so "$SITE_PACKAGES/dgl/libdgl.so"
```

## 4. Install MatGL
Install MatGL without pulling in conflicting dependencies:
```bash
pip install matgl --no-deps
```
*Validated Version:* `2.0.4`

## 5. Benchmarking Results
Verified using `benchmark_gpu_models.py` (located in `.agent/test/benchmark_gpu/`).

| Model | Status | Device | Performance (50 steps MD) |
|---|---|---|---|
| **TensorNetDGL** | **SUCCESS** | `cuda:0` | ~3.1s |
| **M3GNet** | **SUCCESS** | `cuda:0` | ~2.4s |
| **CHGNet** | **SUCCESS** | `cuda:0` | ~2.2s |

All models successfully perform Molecular Dynamics simulation on the GPU.

## 6. Maintenance
- **Prevention:** Always use `--no-deps` when installing wrappers around torch/dgl to avoid overwriting the custom build.
- **Hygiene:** Temporary files should be directed to `.agent/test/`.
