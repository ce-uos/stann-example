cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
find_package(XRT REQUIRED)
find_package(OpenMP)
find_package(OpenCV)

if(NOT XRT_FOUND)
  MESSAGE(FATAL_ERROR "XRT not found, please source XRT setup script before building")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hw_src/stann_inference_lenet.xclbin ${CMAKE_CURRENT_BINARY_DIR}/stann_inference_lenet.xclbin COPYONLY)

# Library of utility functions common to all applications
add_library(example_utils STATIC
  sw_src/xilinx_helpers/xilinx_ocl_helper.cpp
)

target_include_directories(example_utils PUBLIC
  ${XRT_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

# OpenCL buffer map example
add_executable(stann_inference
  sw_src/stann_inference.cpp
  )

target_include_directories(stann_inference PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/sw_src
  ${CMAKE_CURRENT_SOURCE_DIR}/sw_src/xilinx_helpers
  ${XRT_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
  )

target_link_libraries(stann_inference PRIVATE
  ${XRT_LIBS}
  ${OpenCL_LIBRARIES}
  pthread
  uuid
  ${CMAKE_DL_LIBS}
  example_utils
  )

