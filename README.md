# Example Project for STANN

This repository contains the LeNet and TenNet examples for the STANN library. 

## Requirements

The example project in this repo is implemented for the Xilinx Alveo U50.

Xilinx Vitis_HLS and Vivado version 2020.2 are needed for the hardware design,
and the Alveo U50 platform files need to be installed in /opt/xilinx/platforms.
For the software application, you also need to have Xilinx XRT installed.

## How to build the example project

First clone the repository:

    git clone --recurse-submodules git@github.com:ce-uos/stann-example.git

After sourcing the settings files for Vitis_HLS 2020.2 and Vivado 2020.2, build the hardware design:

    source /tools/Xilinx/Vitis_HLS/2020.2/settings64.sh
    source /tools/Xilinx/Vivado/2020.2/settings64.sh
    cd hw_src
    make all

The make target "all" will build the LeNet example design.
After sourcing the setup files for Xilinx XRT, build the software application:
    
    source /opt/xilinx/xrt/setup.sh
    cd ../sw_src
    mkdir build 
    cd build
    cmake ../..
    make

## C Simulation

If you want to run the C testbench for debugging, use the following commands:

    cd hw_src
    make tb

## Citation

If you use this repository, please cite this paper:

    @InProceedings{stann2023,
        author="Rothmann, Marc and Porrmann, Mario",
        title="STANN - Synthesis Templates for Artificial Neural Network Inference and Training",
        booktitle="17th International Work-Conference on Artificial Neural Networks (IWANN2023)",
        year="2023",
        publisher="Springer International Publishing"
    }
