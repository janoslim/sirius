target_sources(sirius_objects PRIVATE flowcsd_config.cpp flowcsd_stats.cpp csd_fpga_runtime.cpp)

# Key Idea 1: the SmartSSD FPGA runtime uses the XRT native C++ API (mango: /opt/xilinx/xrt). Only that source sees
# the XRT headers; sirius-targets.cmake links libxrt_coreutil and adds its directory to the library run path.
set(FLOWCSD_XRT_DIR "/opt/xilinx/xrt" CACHE PATH "XRT install used by the SmartSSD CSD runtime")
find_library(FLOWCSD_XRT_COREUTIL xrt_coreutil PATHS "${FLOWCSD_XRT_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
set_source_files_properties(csd_fpga_runtime.cpp TARGET_DIRECTORY sirius_objects
                            PROPERTIES INCLUDE_DIRECTORIES "${FLOWCSD_XRT_DIR}/include")
