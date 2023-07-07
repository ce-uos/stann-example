#include <stdio.h>
#include <string.h>
#include <cstdlib>

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "../stann_examples.hpp"
#include "../LeNet_inputs.hpp"
#include "../lenet_weights.hpp"

#define IN_BUFSIZE (32*32)
#define OUT_BUFSIZE (10)
#define PARAMS_BUFSIZE (61706)
#define WEIGHTS_BUFSIZE (61470)
#define BIASES_BUFSIZE (236)

template <int OFFSET, int OC, int IC, int K>
void copy_filters(float *input, unsigned int *output) {
    for (int oc = 0; oc < OC; oc++) {
        for (int ic = 0; ic < IC; ic++) {
            for (int y = 0; y < K; y++) {
                for (int x = 0; x < K; x++) {
                    int idx1 = OFFSET + (oc * IC * K * K) + (ic * K * K) + (y * K) + x;
                    int idx2 = OFFSET + (oc * IC * K * K) + (ic * K * K) + (y * K) + x;
                    output[idx1] = *reinterpret_cast<unsigned int *>(&input[idx2]);
                }
            }
        }
    }
}

int main() {
    printf("TEST SIMUL\n");
    unsigned int input[IN_BUFSIZE];
    unsigned int output[OUT_BUFSIZE];
    unsigned int axi_params[PARAMS_BUFSIZE];
    float tmp = 1.0f;

    for (int i = 0; i < IN_BUFSIZE; i++) {
        //input[i] = *reinterpret_cast<unsigned int *>(&tmp);
        input[i] = *reinterpret_cast<unsigned int *>(&lenet_example_input[i]);
    }
    //for (int x = 0; x < 32; x++) {
    //    for (int y = 0; y < 32; y++) {
    //        input[y * 32 + x] = *reinterpret_cast<unsigned int *>(&lenet_example_input[x * 32 + y]);
    //    }
    //}

    // conv1
    //copy_filters<0,6,1,5>(LeNet::weights, axi_params);

    // conv2
    //copy_filters<150,16,6,5>(LeNet::weights, axi_params);


    // rest
    for (int i = 0; i < WEIGHTS_BUFSIZE; i++) {
        axi_params[i] = *reinterpret_cast<unsigned int *>(&LeNet::weights[i]);
    }

    for (int i = 0; i < BIASES_BUFSIZE; i++) {
        axi_params[WEIGHTS_BUFSIZE+i] = *reinterpret_cast<unsigned int *>(&LeNet::biases[i]);
    }

    stann_inference_lenet(input, output, axi_params, 1);

    float foutput[10];

    for (int i = 0; i < 10; i++) {
        foutput[i] = *reinterpret_cast<float*>(&output[i]);
    }

    for (int i = 0; i < 10; i++) {
        printf("%f\n", foutput[i]);
    }
    
}
