# =============================================================================
# flowcsd_tutti_bridge -- Tutti's userspace runtime behind a plain C++ API.
#
# Builds Tutti from source (runtime, local-file resolver, local-NVMe DataPath,
# daemon client; no snvme kernel module, no tests) inside the same pixi
# environment as Sirius, so the gRPC, protobuf, yaml-cpp and CUDA runtime it
# links are the libraries the Sirius process already loads.
#
# Installing the `flowcsd_tutti_bridge` component puts libflowcsd_tutti_bridge.so,
# flowcsd_tutti_bridge.hpp and Tutti's two shared libraries (libnvm.so,
# libnvmeservice.so) into ONE directory; Sirius links the bridge when
# SIRIUS_TUTTI_BRIDGE_DIR names it.  Driven by
# motivational_studies/storage-gpu-query/sirius/03_build_tutti_bridge.sh.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# Tutti appends `-gencode arch=compute_<arch>,code=sm_<arch>` for every entry,
# so this must be a plain number (86 = the RTX A4000), not "86-real".  Set before
# project() so the CUDAARCHS list exported by the pixi environment does not win.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture of the GPU Tutti drives")

project(flowcsd_tutti_bridge LANGUAGES C CXX CUDA)

set(TUTTI_ROOT "$ENV{HOME}/src/tutti" CACHE PATH "Tutti repository to build")

# Only what a Tutti client needs.  Normal variables: Tutti's option() calls
# honour them (CMP0077).
set(TUTTI_FEATURE_LOCAL_NVME ON)
set(TUTTI_BUILD_KERNEL_MODULE OFF) # snvme.ko is built and loaded by the setup scripts
set(TUTTI_FEATURE_MEMFS_SAMPLE OFF)
set(BUILD_TESTING OFF)
# Tutti's static libraries end up inside a shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# The shared libcudart of the pixi environment, as libcudf uses.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

add_subdirectory("${TUTTI_ROOT}" tutti EXCLUDE_FROM_ALL)

find_package(CUDAToolkit REQUIRED)

add_library(flowcsd_tutti_bridge SHARED flowcsd_tutti_bridge.cpp)
# CUDA::nvtx3 is header-only (NVTX ranges for the FlowCSD trace; no-ops without a profiler).
target_link_libraries(flowcsd_tutti_bridge PRIVATE tutti_runtime CUDA::cudart CUDA::nvtx3)
# Tutti's public cuda_like.h includes <cuda/atomic>; CUDA 13 keeps CCCL under
# <toolkit include>/cccl, which Tutti adds only inside its own tree (as the
# tutti_probe build does, the bridge adds it itself).
foreach(include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
  if(EXISTS "${include_dir}/cccl")
    target_include_directories(flowcsd_tutti_bridge PRIVATE "${include_dir}/cccl")
  endif()
endforeach()
# Keep the symbols of Tutti's static libraries out of the bridge's dynamic
# symbol table, so nothing else loaded into Sirius can bind to them.
target_link_options(flowcsd_tutti_bridge PRIVATE "LINKER:--exclude-libs,ALL")
set_target_properties(
  flowcsd_tutti_bridge
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # $ORIGIN finds libnvm.so / libnvmeservice.so next to the bridge;
             # the pixi lib directory finds gRPC, protobuf, yaml-cpp, cudart.
             INSTALL_RPATH "$ORIGIN;$ENV{CONDA_PREFIX}/lib")

# One flat directory: the bridge, its header and the Tutti shared libraries it
# needs.  A named component, because Tutti's own install rules (headers, static
# archives, daemon) are still generated for the EXCLUDE_FROM_ALL subdirectory
# and are not wanted here.
install(
  TARGETS flowcsd_tutti_bridge libnvm nvmeservice
  LIBRARY DESTINATION . COMPONENT flowcsd_tutti_bridge)
install(
  FILES flowcsd_tutti_bridge.hpp
  DESTINATION .
  COMPONENT flowcsd_tutti_bridge)
