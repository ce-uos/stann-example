# Options for TARGET: sw_emu, hw_emu and hw
HW := alveo
TARGET := hw #hw

DEVICE := u50
PLATFORM := xilinx_$(DEVICE)_gen3x16_xdma_201920_3
PLATFORM_REPO_PATHS := /opt/xilinx/platforms/
PFM := $(PLATFORM_REPO_PATHS)/$(PLATFORM)/$(PLATFORM).xpfm

NCPUS := $(shell grep -c ^processor /proc/cpuinfo)
JOBS := $(shell expr $(NCPUS) - 1)

VPPFLAGS := --platform $(PFM) -t $(TARGET) -s -g --config profile.ini
VPPLFLAGS := --jobs $(JOBS)

STANN_FILES += stann/stann.hpp
STANN_FILES += stann/activations.hpp
STANN_FILES += stann/convlayer.hpp
STANN_FILES += stann/convlayer_im2row.hpp
STANN_FILES += stann/convlayer_kn2row.hpp
STANN_FILES += stann/convlayer_winograd.hpp
STANN_FILES += stann/denselayer.hpp
STANN_FILES += stann/denselayer_stream.hpp
STANN_FILES += stann/loss.hpp
STANN_FILES += stann/matmul.hpp
STANN_FILES += stann/padding.hpp
STANN_FILES += stann/poolinglayers.hpp
STANN_FILES += stann/streamutils.hpp
STANN_FILES += stann/utils.hpp

LENET_XOS = stann_inference_lenet.xo
TENNET_XOS = stann_inference_tennet.xo 
TRAINING_XOS = stann_training_tennet.xo 

IP_CACHE_DIR ?= ../../../../ip_cache

.phony: clean traces help

all: stann_inference_lenet.xclbin

kernels: $(LENET_XOS) $(TENNET_XOS) $(TRAINING_XOS)

stann_training_tennet.xclbin: $(TRAINING_XOS) $(BOARD_CONFIG) 
	v++ -l $(VPPFLAGS) $(VPPLFLAGS) -o $@ $(TRAINING_XOS) --remote_ip_cache ${IP_CACHE_DIR}

stann_inference_tennet.xclbin: $(TENNET_XOS) $(BOARD_CONFIG) 
	v++ -l $(VPPFLAGS) $(VPPLFLAGS) -o $@ $(TENNET_XOS) --remote_ip_cache ${IP_CACHE_DIR}

stann_inference_lenet.xclbin: $(LENET_XOS) $(BOARD_CONFIG) 
	v++ -l $(VPPFLAGS) $(VPPLFLAGS) -o $@ $(LENET_XOS) --remote_ip_cache ${IP_CACHE_DIR}
	
stann_inference_tennet.xo: stann_examples.cpp $(STANN_FILES)
	v++ --kernel stann_inference_tennet $(VPPFLAGS) -c -o $@ $<

stann_inference_lenet.xo: stann_examples.cpp $(STANN_FILES)
	v++ --kernel stann_inference_lenet $(VPPFLAGS) -c -o $@ $<

stann_training_tennet.xo: stann_examples.cpp $(STANN_FILES)
	v++ --kernel stann_training_tennet $(VPPFLAGS) -c -o $@ $<

clean:
	$(RM) -r *.xo _x .Xil sd_card *.xclbin *.ltx *.log *.info *compile_summary* vitis_analyzer* *link_summary*
