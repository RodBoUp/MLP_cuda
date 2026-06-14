#pragma once

#include "gpu_layer.h"
#include "../data/mnist_loader.h"

class GPUMLP
{
private:
    static constexpr int BATCH_SIZE = 256;

    GPULayer layer1;
    GPULayer layer2;
    GPULayer layer3;

    float* d_batchInput = nullptr;
    int* d_batchLabels = nullptr;

    float* d_hidden1 = nullptr;
    float* d_hidden2 = nullptr;
    float* d_output = nullptr;

    float* d_target = nullptr;

    float* d_errorOutput = nullptr;
    float* d_errorHidden2 = nullptr;
    float* d_errorHidden1 = nullptr;

    float* d_trainImages = nullptr;
    int* d_trainLabels = nullptr;

public:

    GPUMLP();
    ~GPUMLP();

    void uploadDataset(
        const std::vector<float>& images,
        const std::vector<int>& labels
    );

    void copyToGPU();

    void forwardGPU(
        const float* batchInput,
        int batchSize
    );

    void predictGPU(
        const float* input,
        float* output,
        int batchSize
    );

    int predictClass(
        const float* input
    );

    void trainBatchFromGPU(
        int batchStart,
        int batchSize,
        float learningRate
    );
};