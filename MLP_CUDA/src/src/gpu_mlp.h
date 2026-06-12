#pragma once

#include "gpu_layer.h"

class GPUMLP
{
private:

    GPULayer layer1;
    GPULayer layer2;
    GPULayer layer3;

    // Activaciones temporales GPU
    float* d_hidden1;
    float* d_hidden2;
    float* d_output;

    // Entrada persistente
    float* d_input;

    // Vector one-hot esperado
    float* d_target;

    // Backpropagation
    float* d_errorOutput;
    float* d_errorHidden2;
    float* d_errorHidden1;

public:

    GPUMLP();

    ~GPUMLP();

    void copyToGPU();

    void forwardGPU(
        const float* d_input
    );

    void predictGPU(
        const float* input,
        float* output
    );
    int predictClass(
    const float* input);


    void trainGPU(
    const float* input,
    int label,
    float learningRate
);

};