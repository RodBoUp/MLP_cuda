#pragma once

struct GPULayer
{
    int inputSize;
    int outputSize;

    // Host (CPU)
    float* weights;
    float* biases;

    // Device (GPU)
    float* d_weights;
    float* d_biases;

    GPULayer();

    GPULayer(
        int inputs,
        int outputs
    );

    ~GPULayer();

    void allocateGPU();

    void freeGPU();

    void copyToGPU();

    void copyFromGPU();
};