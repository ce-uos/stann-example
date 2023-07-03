#include "ap_int.h"

#include "STANN/stann.hpp"
#include "constants.h"
#include "tennet_weights.h"

typedef float stann_value_t;

struct TenNetParams {
    stann_value_t weights_l1[PBNET_HIDDEN_NEURONS*6];
    stann_value_t weights_l2[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l3[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l4[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l5[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l6[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l7[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l8[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l9[PBNET_HIDDEN_NEURONS*PBNET_HIDDEN_NEURONS];
    stann_value_t weights_l10[8*PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l1[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l2[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l3[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l4[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l5[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l6[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l7[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l8[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l9[PBNET_HIDDEN_NEURONS];
    stann_value_t biases_l10[8];
};

struct LeNetParams {
    stann_value_t weights_l1[150];
    stann_value_t weights_l2[2400];
    stann_value_t weights_l3[400*120];
    stann_value_t weights_l4[120*84];
    stann_value_t weights_l5[84*10];
    stann_value_t biases_l1[6];
    stann_value_t biases_l2[16];
    stann_value_t biases_l3[120];
    stann_value_t biases_l4[84];
    stann_value_t biases_l5[10];
};

namespace LeNet {

void forward(stann_value_t *input, LeNetParams &params, stann_value_t *output, int reps) {
#pragma HLS dataflow
    hls::stream<float> input_stream;
    hls::stream<float> conv1_out;
    hls::stream<float> pool1_out;
    hls::stream<float> conv2_out;
    hls::stream<float> pool2_out;
    hls::stream<float> dense1_out;
    hls::stream<float> dense2_out;
    hls::stream<float> dense3_out;

    StreamUtil::tostream<32*32>(input, input_stream, reps);

    // The following lines could be used for a kn2row implementation:
    //
    // ConvLayer::kn2row::Float::forward<32,32,1,6,5,2,1,8>(input_stream, params.weights_l1, params.biases_l1, conv1_out, LEAKY_RELU, reps);
    // PoolingLayer::average_stream<28,28,6,2>(conv1_out, pool1_out, reps);
    // ConvLayer::kn2row::Float::forward<14,14,6,16,5,2,8,4>(pool1_out, params.weights_l2, params.biases_l2, conv2_out, LEAKY_RELU, reps);
    // PoolingLayer::average_stream<10,10,16,2>(conv2_out, pool2_out, reps);

    // The following lines are for the im2row implementation. Remove if the
    // kn2row implementation should be used.
    ConvLayer::im2row::Float::forward<32,32,1,6,5,1,5,4>(input_stream, params.weights_l1, params.biases_l1, conv1_out, LEAKY_RELU, reps);
    PoolingLayer::average_stream<28,28,6,2>(conv1_out, pool1_out, reps);
    ConvLayer::im2row::Float::forward<14,14,6,16,5,1,5,4>(pool1_out, params.weights_l2, params.biases_l2, conv2_out, LEAKY_RELU, reps);
    PoolingLayer::average_stream<10,10,16,2>(conv2_out, pool2_out, reps);

    DenseLayerStream::Float::forward<400,120,8,8,1>(pool2_out, params.weights_l3, params.biases_l3, dense1_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<120,84,4,4,1>(dense1_out, params.weights_l4, params.biases_l4, dense2_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<84,10,2,2,1>(dense2_out, params.weights_l5, params.biases_l5, dense3_out, NONE, reps);

    StreamUtil::toarray<10>(dense3_out, output, reps);

}

} // namespace LeNet

namespace TenNet {

void triplicate_params(TenNetParams &params, TenNetParams &out_params1, TenNetParams &out_params2) {
    for (int i = 0; i < 6 * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l1[i] = params.weights_l1[i];
        out_params2.weights_l1[i] = params.weights_l1[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l2[i] = params.weights_l2[i];
        out_params2.weights_l2[i] = params.weights_l2[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l3[i] = params.weights_l3[i];
        out_params2.weights_l3[i] = params.weights_l3[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l4[i]= params.weights_l4[i];
        out_params2.weights_l4[i]= params.weights_l4[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l5[i] = params.weights_l5[i];
        out_params2.weights_l5[i] = params.weights_l5[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l6[i] = params.weights_l6[i];
        out_params2.weights_l6[i] = params.weights_l6[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l7[i] = params.weights_l7[i];
        out_params2.weights_l7[i] = params.weights_l7[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l8[i] = params.weights_l8[i];
        out_params2.weights_l8[i] = params.weights_l8[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l9[i] = params.weights_l9[i];
        out_params2.weights_l9[i] = params.weights_l9[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS * 8; i++) {
    #pragma HLS pipeline II=3
        out_params1.weights_l10[i] = params.weights_l10[i];
        out_params2.weights_l10[i] = params.weights_l10[i];
    }

    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l1[i] = params.biases_l1[i];
        out_params2.biases_l1[i] = params.biases_l1[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l2[i] = params.biases_l2[i];
        out_params2.biases_l2[i] = params.biases_l2[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l3[i] = params.biases_l3[i];
        out_params2.biases_l3[i] = params.biases_l3[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l4[i]= params.biases_l4[i];
        out_params2.biases_l4[i]= params.biases_l4[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l5[i] = params.biases_l5[i];
        out_params2.biases_l5[i] = params.biases_l5[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l6[i] = params.biases_l6[i];
        out_params2.biases_l6[i] = params.biases_l6[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l7[i] = params.biases_l7[i];
        out_params2.biases_l7[i] = params.biases_l7[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l8[i] = params.biases_l8[i];
        out_params2.biases_l8[i] = params.biases_l8[i];
    }
    for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l9[i] = params.biases_l9[i];
        out_params2.biases_l9[i] = params.biases_l9[i];
    }
    for (int i = 0; i < 8; i++) {
    #pragma HLS pipeline II=3
        out_params1.biases_l10[i] = params.biases_l10[i];
        out_params2.biases_l10[i] = params.biases_l10[i];
    }
}

template<int BATCH_SIZE>
void training(stann_value_t *input, stann_value_t *labels,
              TenNetParams &fw_params, TenNetParams &bw_params, TenNetParams &up_params,
              stann_value_t learning_rate, int reps) {
              #pragma HLS disaggregate variable=fw_params
              #pragma HLS disaggregate variable=bw_params
              #pragma HLS disaggregate variable=up_params
              #pragma HLS Dataflow
    hls::stream<stann_value_t> input_stream;
    hls::stream<stann_value_t> input_stream1;
    hls::stream<stann_value_t> input_stream2;
    hls::stream<stann_value_t> output_stream;

    hls::stream<stann_value_t> l1_out;
    hls::stream<stann_value_t> l2_out;
    hls::stream<stann_value_t> l3_out;
    hls::stream<stann_value_t> l4_out;
    hls::stream<stann_value_t> l5_out;
    hls::stream<stann_value_t> l6_out;
    hls::stream<stann_value_t> l7_out;
    hls::stream<stann_value_t> l8_out;
    hls::stream<stann_value_t> l9_out;
    hls::stream<stann_value_t> l10_out;

    float l1_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l2_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l3_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l4_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l5_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l6_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l7_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l8_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];
    float l9_out_copy[PBNET_HIDDEN_NEURONS * BATCH_SIZE];

    hls::stream<stann_value_t> l1_out_relu;
    hls::stream<stann_value_t> l2_out_relu;
    hls::stream<stann_value_t> l3_out_relu;
    hls::stream<stann_value_t> l4_out_relu;
    hls::stream<stann_value_t> l5_out_relu;
    hls::stream<stann_value_t> l6_out_relu;
    hls::stream<stann_value_t> l7_out_relu;
    hls::stream<stann_value_t> l8_out_relu;
    hls::stream<stann_value_t> l9_out_relu;
    hls::stream<stann_value_t> l10_out_relu;

    hls::stream<stann_value_t> l1_out_relu_copy;
    hls::stream<stann_value_t> l2_out_relu_copy;
    hls::stream<stann_value_t> l3_out_relu_copy;
    hls::stream<stann_value_t> l4_out_relu_copy;
    hls::stream<stann_value_t> l5_out_relu_copy;
    hls::stream<stann_value_t> l6_out_relu_copy;
    hls::stream<stann_value_t> l7_out_relu_copy;
    hls::stream<stann_value_t> l8_out_relu_copy;
    hls::stream<stann_value_t> l9_out_relu_copy;
    hls::stream<stann_value_t> l10_out_copy1;
    hls::stream<stann_value_t> l10_out_copy2;

    StreamUtil::tostream<6>(input, input_stream, reps);
    StreamUtil::duplicate<6*BATCH_SIZE>(input_stream, input_stream1, input_stream2);

    DenseLayerStream::Float::forward<6, PBNET_HIDDEN_NEURONS, 2, 2, 1, 80>(input_stream1, fw_params.weights_l1,  fw_params.biases_l1, l1_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l1_out, l1_out_copy, l1_out_relu, l1_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l1_out_relu, fw_params.weights_l2,  fw_params.biases_l2, l2_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l2_out, l2_out_copy, l2_out_relu, l2_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l2_out_relu, fw_params.weights_l3,  fw_params.biases_l3, l3_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l3_out, l3_out_copy, l3_out_relu, l3_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l3_out_relu, fw_params.weights_l4,  fw_params.biases_l4, l4_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l4_out, l4_out_copy, l4_out_relu, l4_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l4_out_relu, fw_params.weights_l5,  fw_params.biases_l5, l5_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l5_out, l5_out_copy, l5_out_relu, l5_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l5_out_relu, fw_params.weights_l6,  fw_params.biases_l6, l6_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l6_out, l6_out_copy, l6_out_relu, l6_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l6_out_relu, fw_params.weights_l7,  fw_params.biases_l7, l7_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l7_out, l7_out_copy, l7_out_relu, l7_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l7_out_relu, fw_params.weights_l8,  fw_params.biases_l8, l8_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l8_out, l8_out_copy, l8_out_relu, l8_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, FW_PE1, FW_PE2, FW_PE3, 80>(l8_out_relu, fw_params.weights_l9,  fw_params.biases_l9, l9_out, NONE, reps);
    ActivationLayer::Float::leaky_relu_stream<PBNET_HIDDEN_NEURONS,BATCH_SIZE>(l9_out, l9_out_copy, l9_out_relu, l9_out_relu_copy, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, 8, FW_PE1, FW_PE2, FW_PE3, 80>(l9_out_relu, fw_params.weights_l10, fw_params.biases_l10, l10_out, NONE, reps);
    StreamUtil::duplicate<8 * BATCH_SIZE>(l10_out, l10_out_copy1, l10_out_copy2);

    hls::stream<float> full_labels_stream;

    hls::stream<float> l1_deltas;
    hls::stream<float> l2_deltas;
    hls::stream<float> l3_deltas;
    hls::stream<float> l4_deltas;
    hls::stream<float> l5_deltas;
    hls::stream<float> l6_deltas;
    hls::stream<float> l7_deltas;
    hls::stream<float> l8_deltas;
    hls::stream<float> l9_deltas;
    hls::stream<float> l10_deltas;

    hls::stream<float> l1_deltas_up;
    hls::stream<float> l2_deltas_up;
    hls::stream<float> l3_deltas_up;
    hls::stream<float> l4_deltas_up;
    hls::stream<float> l5_deltas_up;
    hls::stream<float> l6_deltas_up;
    hls::stream<float> l7_deltas_up;
    hls::stream<float> l8_deltas_up;
    hls::stream<float> l9_deltas_up;
    hls::stream<float> l10_deltas_up;

    hls::stream<float> l1_deltas_bw;
    hls::stream<float> l2_deltas_bw;
    hls::stream<float> l3_deltas_bw;
    hls::stream<float> l4_deltas_bw;
    hls::stream<float> l5_deltas_bw;
    hls::stream<float> l6_deltas_bw;
    hls::stream<float> l7_deltas_bw;
    hls::stream<float> l8_deltas_bw;
    hls::stream<float> l9_deltas_bw;
    hls::stream<float> l10_deltas_bw;

    StreamUtil::tostream<8>(labels, full_labels_stream, reps);
    Loss::MeanSquaredError_derivative_stream<8, BATCH_SIZE>(l10_out_copy2, full_labels_stream, l10_deltas);

    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l10_deltas, l10_deltas_bw, l10_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l9_out_copy, bw_params.weights_l9, l10_deltas_bw, l9_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l9_deltas, l9_deltas_bw, l9_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l8_out_copy, bw_params.weights_l8, l9_deltas_bw, l8_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l8_deltas, l8_deltas_bw, l8_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l7_out_copy, bw_params.weights_l7, l8_deltas_bw, l7_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l7_deltas, l7_deltas_bw, l7_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l6_out_copy, bw_params.weights_l6, l7_deltas_bw, l6_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l6_deltas, l6_deltas_bw, l6_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l5_out_copy, bw_params.weights_l5, l6_deltas_bw, l5_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l5_deltas, l5_deltas_bw, l5_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l4_out_copy, bw_params.weights_l4, l5_deltas_bw, l4_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l4_deltas, l4_deltas_bw, l4_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l3_out_copy, bw_params.weights_l3, l4_deltas_bw, l3_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l3_deltas, l3_deltas_bw, l3_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,BATCH_SIZE,BW_PE1, BW_PE2, BW_PE3, 80>(l2_out_copy, bw_params.weights_l2, l3_deltas_bw, l2_deltas, LEAKY_RELU, reps);
    StreamUtil::duplicate<PBNET_HIDDEN_NEURONS * BATCH_SIZE>(l2_deltas, l2_deltas_bw, l2_deltas_up);
    DenseLayerStream::Float::backward<PBNET_HIDDEN_NEURONS,PBNET_HIDDEN_NEURONS,8,BATCH_SIZE,8,8,8, 80>(l1_out_copy, bw_params.weights_l1, l2_deltas_bw, l1_deltas_up, LEAKY_RELU, reps);


    DenseLayerStream::Float::update<6, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l1_deltas_up, up_params.weights_l1, up_params.biases_l1, input_stream2, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l2_deltas_up, up_params.weights_l2, up_params.biases_l2, l1_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l3_deltas_up, up_params.weights_l3, up_params.biases_l3, l2_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l4_deltas_up, up_params.weights_l4, up_params.biases_l4, l3_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l5_deltas_up, up_params.weights_l5, up_params.biases_l5, l4_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l6_deltas_up, up_params.weights_l6, up_params.biases_l6, l5_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l7_deltas_up, up_params.weights_l7, up_params.biases_l7, l6_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l8_deltas_up, up_params.weights_l8, up_params.biases_l8, l7_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l9_deltas_up, up_params.weights_l9, up_params.biases_l9, l8_out_relu_copy, learning_rate);
    DenseLayerStream::Float::update<PBNET_HIDDEN_NEURONS, 8, BATCH_SIZE, float, UP_PE1, UP_PE2, UP_PE3>(l10_deltas_up, up_params.weights_l10, up_params.biases_l10, l9_out_relu_copy, learning_rate);

}

void forward(stann_value_t *input, TenNetParams &params,
             stann_value_t *output, int reps) {
             #pragma HLS Dataflow

    hls::stream<stann_value_t> input_stream;
    hls::stream<stann_value_t> output_stream;

    hls::stream<stann_value_t> l1_out;
    hls::stream<stann_value_t> l2_out;
    hls::stream<stann_value_t> l3_out;
    hls::stream<stann_value_t> l4_out;
    hls::stream<stann_value_t> l5_out;
    hls::stream<stann_value_t> l6_out;
    hls::stream<stann_value_t> l7_out;
    hls::stream<stann_value_t> l8_out;
    hls::stream<stann_value_t> l9_out;
    hls::stream<stann_value_t> l10_out;

    StreamUtil::tostream<6>(input, input_stream, reps);

    DenseLayerStream::Float::forward<6, PBNET_HIDDEN_NEURONS,   2, INF_PE2, 1>(input_stream,    params.weights_l1,  params.biases_l1, l1_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l1_out, params.weights_l2,  params.biases_l2, l2_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l2_out, params.weights_l3,  params.biases_l3, l3_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l3_out, params.weights_l4,  params.biases_l4, l4_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l4_out, params.weights_l5,  params.biases_l5, l5_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l5_out, params.weights_l6,  params.biases_l6, l6_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l6_out, params.weights_l7,  params.biases_l7, l7_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l7_out, params.weights_l8,  params.biases_l8, l8_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, PBNET_HIDDEN_NEURONS, INF_PE1, INF_PE2, INF_PE3>(l8_out, params.weights_l9,  params.biases_l9, l9_out, LEAKY_RELU, reps);
    DenseLayerStream::Float::forward<PBNET_HIDDEN_NEURONS, 8,   INF_PE1, 2, 1>(l9_out, params.weights_l10, params.biases_l10, output_stream, NONE, reps);

    StreamUtil::toarray<8>(output_stream, output, reps);
}

} // namespace TenNet

extern "C" {

    void stann_inference_lenet(unsigned int *inputs_int, unsigned int *outputs_int, unsigned int *axi_params, int mode) {
    #pragma HLS INTERFACE m_axi port=inputs_int offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=outputs_int offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=axi_params offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=inputs_int bundle=control
    #pragma HLS INTERFACE s_axilite port=outputs_int bundle=control
    #pragma HLS INTERFACE s_axilite port=axi_params bundle=control
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control


        stann_value_t inputs[(32*32) * MINI_BATCH_SIZE];
        stann_value_t outputs[10 * MINI_BATCH_SIZE];

        for (int i = 0; i < (32*32) * MINI_BATCH_SIZE; i++) {
            inputs[i] = *reinterpret_cast<float*>(&inputs_int[i]);
        }

        static LeNetParams params;
        #pragma HLS bind_storage variable=params.weights_l1 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l2 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l3 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l4 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l5 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l1 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l2 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l3 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l4 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l5 type=ram_t2p impl=BRAM

        if (mode == 1) {
            // read weights
            for (int i = 0; i < 150; i++) {
            #pragma HLS pipeline II=3
                params.weights_l1[i] = axi_params[i];
            }
            for (int i = 0; i < 2400; i++) {
            #pragma HLS pipeline II=3
                params.weights_l2[i] = axi_params[150+i];
            }
            for (int i = 0; i < 48000; i++) {
            #pragma HLS pipeline II=3
                params.weights_l3[i] = axi_params[2550+i];
            }
            for (int i = 0; i < 10080; i++) {
            #pragma HLS pipeline II=3
                params.weights_l4[i] = axi_params[50550+i];
            }
            for (int i = 0; i < 840; i++) {
            #pragma HLS pipeline II=3
                params.weights_l5[i] = axi_params[60630+i];
            }
            for (int i = 0; i < 6; i++) {
            #pragma HLS pipeline II=3
                params.biases_l1[i] = axi_params[61470+i];
            }
            for (int i = 0; i < 16; i++) {
            #pragma HLS pipeline II=3
                params.biases_l2[i] = axi_params[61470+6+i];
            }
            for (int i = 0; i < 120; i++) {
            #pragma HLS pipeline II=3
                params.biases_l3[i] = axi_params[61470+22+i];
            }
            for (int i = 0; i < 84; i++) {
            #pragma HLS pipeline II=3
                params.biases_l4[i] = axi_params[61470+142+i];
            }
            for (int i = 0; i < 10; i++) {
            #pragma HLS pipeline II=3
                params.biases_l5[i] = axi_params[61470+226+i];
            }
        }

        LeNet::forward(inputs, params, outputs, MINI_BATCH_SIZE);

        for (int i = 0; i < 10 * MINI_BATCH_SIZE; i++) {
            outputs_int[i] = *reinterpret_cast<unsigned int*>(&outputs[i]);
        }
    }

    void stann_inference_tennet(unsigned int *inputs_int, unsigned int *outputs_int) {
    #pragma HLS INTERFACE m_axi port=inputs_int offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=outputs_int offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=inputs_int bundle=control
    #pragma HLS INTERFACE s_axilite port=outputs_int bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

        stann_value_t inputs[6 * MINI_BATCH_SIZE];
        stann_value_t outputs[8 * MINI_BATCH_SIZE];

        for (int i = 0; i < 6 * MINI_BATCH_SIZE; i++) {
            inputs[i] = *reinterpret_cast<float*>(&inputs_int[i]);
        }

        // initialize with constants from tennet_weights.h
        static TenNetParams params = {
            PBWEIGHTS_INIT_1,
            PBWEIGHTS_INIT_2,
            PBWEIGHTS_INIT_3,
            PBWEIGHTS_INIT_4,
            PBWEIGHTS_INIT_5,
            PBWEIGHTS_INIT_5,
            PBWEIGHTS_INIT_6,
            PBWEIGHTS_INIT_7,
            PBWEIGHTS_INIT_9,
            PBWEIGHTS_INIT_10,
            PBBIASES_INIT_1,
            PBBIASES_INIT_2,
            PBBIASES_INIT_3,
            PBBIASES_INIT_4,
            PBBIASES_INIT_5,
            PBBIASES_INIT_6,
            PBBIASES_INIT_7,
            PBBIASES_INIT_8,
            PBBIASES_INIT_9,
            PBBIASES_INIT_10,
        };
        #pragma HLS bind_storage variable=params.weights_l1 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l2 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l3 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l4 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l5 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l6 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l7 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l8 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l9 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l10 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l1 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l2 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l3 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l4 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l5 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l6 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l7 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l8 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l9 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l10 type=ram_t2p impl=BRAM

        TenNet::forward(inputs, params, outputs, MINI_BATCH_SIZE);

        for (int i = 0; i < 8 * MINI_BATCH_SIZE; i++) {
            outputs_int[i] = *reinterpret_cast<unsigned int*>(&outputs[i]);
        }
    }


    void stann_training_tennet(unsigned int *train_data_int, unsigned int *train_labels_int, unsigned int *axi_weights, unsigned int *axi_biases, unsigned int lr_int, unsigned int axi_mode) {
    #pragma HLS INTERFACE m_axi port=train_data_int offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=train_labels_int offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=axi_weights offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=axi_biases offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=train_data_int bundle=control
    #pragma HLS INTERFACE s_axilite port=train_labels_int bundle=control
    #pragma HLS INTERFACE s_axilite port=axi_weights bundle=control
    #pragma HLS INTERFACE s_axilite port=axi_biases bundle=control
    #pragma HLS INTERFACE s_axilite port=lr_int bundle=control
    #pragma HLS INTERFACE s_axilite port=axi_mode bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

        float learning_rate = *reinterpret_cast<float*>(&lr_int);
        float learning_factor = learning_rate / MINI_BATCH_SIZE * 2;

        static TenNetParams params = {
            PBWEIGHTS_INIT_1,
            PBWEIGHTS_INIT_2,
            PBWEIGHTS_INIT_3,
            PBWEIGHTS_INIT_4,
            PBWEIGHTS_INIT_5,
            PBWEIGHTS_INIT_5,
            PBWEIGHTS_INIT_6,
            PBWEIGHTS_INIT_7,
            PBWEIGHTS_INIT_9,
            PBWEIGHTS_INIT_10,
            PBBIASES_INIT_1,
            PBBIASES_INIT_2,
            PBBIASES_INIT_3,
            PBBIASES_INIT_4,
            PBBIASES_INIT_5,
            PBBIASES_INIT_6,
            PBBIASES_INIT_7,
            PBBIASES_INIT_8,
            PBBIASES_INIT_9,
            PBBIASES_INIT_10,
        };
        #pragma HLS bind_storage variable=params.weights_l1 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l2 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l3 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l4 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l5 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l6 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l7 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l8 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l9 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.weights_l10 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l1 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l2 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l3 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l4 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l5 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l6 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l7 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l8 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l9 type=ram_t2p impl=BRAM
        #pragma HLS bind_storage variable=params.biases_l10 type=ram_t2p impl=BRAM

        TenNetParams fw_params;
        #pragma HLS bind_storage variable=fw_params.weights_l1 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l2 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l3 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l4 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l5 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l6 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l7 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l8 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l9 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.weights_l10 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l1 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l2 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l3 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l4 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l5 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l6 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l7 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l8 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l9 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=fw_params.biases_l10 type=ram_t2p impl=URAM

        TenNetParams bw_params;
        #pragma HLS bind_storage variable=bw_params.weights_l1 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l2 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l3 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l4 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l5 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l6 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l7 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l8 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l9 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.weights_l10 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l1 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l2 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l3 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l4 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l5 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l6 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l7 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l8 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l9 type=ram_t2p impl=URAM
        #pragma HLS bind_storage variable=bw_params.biases_l10 type=ram_t2p impl=URAM

        unsigned int mode = axi_mode;

        if (mode == 0) {
            stann_value_t train_data[MINI_BATCH_SIZE * 6];
            stann_value_t train_labels[MINI_BATCH_SIZE * 8];

            for (int i = 0; i < MINI_BATCH_SIZE * 6; i++) {
            #pragma HLS pipeline II=3
                train_data[i] = *reinterpret_cast<stann_value_t*>(&train_data_int[i]);
            }

            for (int i = 0; i < MINI_BATCH_SIZE * 8; i++) {
            #pragma HLS pipeline II=3
                train_labels[i] = *reinterpret_cast<stann_value_t*>(&train_labels_int[i]);
            }

            TenNet::triplicate_params(params, fw_params, bw_params);

            TenNet::training<MINI_BATCH_SIZE>(train_data, train_labels, fw_params, bw_params, params, learning_factor, MINI_BATCH_SIZE);

        } else if (mode == 1) {
            // write weights
            for (int i = 0; i < 6 * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[i] = params.weights_l1[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 0 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l2[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 1 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l3[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 2 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l4[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 3 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l5[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 4 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l6[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 5 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l7[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 6 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l8[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 7 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l9[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS * 8; i++) {
            #pragma HLS pipeline II=3
                axi_weights[6 * PBNET_HIDDEN_NEURONS + 8 * PBNET_HIDDEN_NEURONS * PBNET_HIDDEN_NEURONS + i] = params.weights_l10[i];
            }

            // write biases
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[0 * PBNET_HIDDEN_NEURONS + i] = params.biases_l1[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[1 * PBNET_HIDDEN_NEURONS + i] = params.biases_l2[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[2 * PBNET_HIDDEN_NEURONS + i] = params.biases_l3[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[3 * PBNET_HIDDEN_NEURONS + i] = params.biases_l4[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[4 * PBNET_HIDDEN_NEURONS + i] = params.biases_l5[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[5 * PBNET_HIDDEN_NEURONS + i] = params.biases_l6[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[6 * PBNET_HIDDEN_NEURONS + i] = params.biases_l7[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[7 * PBNET_HIDDEN_NEURONS + i] = params.biases_l8[i];
            }
            for (int i = 0; i < PBNET_HIDDEN_NEURONS; i++) {
            #pragma HLS pipeline II=3
                axi_biases[8 * PBNET_HIDDEN_NEURONS + i] = params.biases_l9[i];
            }
            for (int i = 0; i < 8; i++) {
            #pragma HLS pipeline II=3
                axi_biases[9 * PBNET_HIDDEN_NEURONS + i] = params.biases_l10[i];
            }
        }


    }
} // extern "C"
