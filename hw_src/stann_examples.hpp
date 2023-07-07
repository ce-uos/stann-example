#ifndef _STANN_EXAMPLES_HPP_
#define _STANN_EXAMPLES_HPP_

extern "C" {
    void stann_inference_lenet(unsigned int *inputs_int, unsigned int *outputs_int, unsigned int *axi_params, int mode);
}

#endif
