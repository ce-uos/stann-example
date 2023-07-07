#include <iostream>
#include <memory>
#include <string>

#include "xilinx_helpers/xilinx_ocl_helper.hpp"

#include "LeNet_inputs.hpp"
#include "lenet_weights.hpp"

#define IN_BUFSIZE (32*32)
#define OUT_BUFSIZE (10)
#define PARAMS_BUFSIZE (61706)
#define WEIGHTS_BUFSIZE (61470)
#define BIASES_BUFSIZE (236)

int main(int argc, char *argv[]) {
    xilinx::example_utils::XilinxOclHelper xocl;
    xocl.initialize("stann_inference_lenet.xclbin");

    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("stann_inference_lenet");

    cl::Buffer in_buf(xocl.get_context(),
                     static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR),
                     IN_BUFSIZE * sizeof(uint32_t),
                     NULL,
                     NULL);

    cl::Buffer out_buf(xocl.get_context(),
                     static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR),
                     OUT_BUFSIZE * sizeof(uint32_t),
                     NULL,
                     NULL);

    cl::Buffer param_buf(xocl.get_context(),
                     static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR),
                     PARAMS_BUFSIZE * sizeof(uint32_t),
                     NULL,
                     NULL);

    krnl.setArg(0, in_buf);
    krnl.setArg(1, out_buf);
    krnl.setArg(2, param_buf);
    krnl.setArg(3, 1);

    uint32_t *input = (uint32_t*) q.enqueueMapBuffer(in_buf, CL_TRUE, CL_MAP_WRITE, 0, IN_BUFSIZE * sizeof(uint32_t));
    for (int i = 0; i < IN_BUFSIZE; i++) {
        input[i] = *reinterpret_cast<unsigned int *>(&lenet_example_input[i]);
    }

    uint32_t *params = (uint32_t*) q.enqueueMapBuffer(param_buf, CL_TRUE, CL_MAP_WRITE, 0, PARAMS_BUFSIZE * sizeof(uint32_t));
    for (int i = 0; i < WEIGHTS_BUFSIZE; i++) {
        params[i] = *reinterpret_cast<unsigned int *>(&LeNet::weights[i]);
    }
    for (int i = 0; i < BIASES_BUFSIZE; i++) {
        params[WEIGHTS_BUFSIZE+i] = *reinterpret_cast<unsigned int *>(&LeNet::biases[i]);
    }

    cl::Event event_sp;
    q.enqueueMigrateMemObjects({in_buf, param_buf}, 0, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    uint32_t *output = (uint32_t*)q.enqueueMapBuffer(out_buf, CL_TRUE, CL_MAP_READ, 0, 10*sizeof(uint32_t));

    for (int i = 0; i < 10; i++) {
        float f = *reinterpret_cast<float*>(&output[i]);
        std::cout << f << std::endl;
    }

    q.enqueueUnmapMemObject(in_buf, input);
    q.enqueueUnmapMemObject(out_buf, output);
    q.enqueueUnmapMemObject(param_buf, params);
    q.finish();

    return 0;
}
