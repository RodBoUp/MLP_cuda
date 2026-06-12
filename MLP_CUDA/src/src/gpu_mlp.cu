#include "gpu_mlp.h"

#include <cuda_runtime.h>
#include <cstdlib>

__global__
void denseForwardKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int inputSize,
    int outputSize
)
{
    int neuron =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(neuron >= outputSize)
        return;

    float sum =
        biases[neuron];

    for(int i = 0; i < inputSize; i++)
    {
        sum +=
            weights[
                neuron * inputSize + i
            ]
            *
            input[i];
    }

    output[neuron] =
        sum > 0.0f ?
        sum :
        0.0f;
}

GPUMLP::GPUMLP()
    :
    layer1(784,128),
    layer2(128,64),
    layer3(64,10) {

    for(int i=0;i<128;i++)
        layer1.biases[i] = 0.0f;

    for(int i=0;i<64;i++)
        layer2.biases[i] = 0.0f;

    for(int i=0;i<10;i++)
        layer3.biases[i] = 0.0f;


    for(int j=0;j<128;j++)
    {
        for(int i=0;i<784;i++)
        {
            layer1.weights[j*784+i] =
                ((float)rand()/RAND_MAX)
                *0.2f
                -0.1f;
        }
    }
    for(int j=0;j<64;j++)
    {
        for(int i=0;i<128;i++)
        {
            layer2.weights[j*128+i] =
                ((float)rand()/RAND_MAX)
                *0.2f
                -0.1f;
        }
    }
    for(int j=0;j<10;j++)
    {
        for(int i=0;i<64;i++)
        {
            layer3.weights[j*64+i] =
                ((float)rand()/RAND_MAX)
                *0.2f
                -0.1f;
        }
    }
    layer1.allocateGPU();
    layer2.allocateGPU();
    layer3.allocateGPU();

    cudaMalloc(
        &d_hidden1,
        128*sizeof(float)
    );

    cudaMalloc(
        &d_hidden2,
        64*sizeof(float)
    );

    cudaMalloc(
        &d_output,
        10*sizeof(float)
    );

    cudaMalloc(
    &d_input,
    784*sizeof(float)
);

    cudaMalloc(
        &d_target,
        10*sizeof(float)
    );

    cudaMalloc(
        &d_errorOutput,
        10*sizeof(float)
    );

    cudaMalloc(
        &d_errorHidden2,
        64*sizeof(float)
    );

    cudaMalloc(
        &d_errorHidden1,
        128*sizeof(float)
    );

}


void GPUMLP::copyToGPU()
{
    layer1.copyToGPU();
    layer2.copyToGPU();
    layer3.copyToGPU();
}

void GPUMLP::forwardGPU(
    const float* d_input
) {
    denseForwardKernel<<<1,128>>>(
    d_input,
    layer1.d_weights,
    layer1.d_biases,
    d_hidden1,
    784,
    128
);

    denseForwardKernel<<<1,64>>>(
    d_hidden1,
    layer2.d_weights,
    layer2.d_biases,
    d_hidden2,
    128,
    64
);

    denseForwardKernel<<<1,10>>>(
    d_hidden2,
    layer3.d_weights,
    layer3.d_biases,
    d_output,
    64,
    10
);
}


void GPUMLP::predictGPU(
    const float* input,
    float* output
)
{
    float* d_input;

    cudaMalloc(
        &d_input,
        784 * sizeof(float)
    );

    cudaMemcpy(
        d_input,
        input,
        784 * sizeof(float),
        cudaMemcpyHostToDevice
    );

    forwardGPU(d_input);

    cudaMemcpy(
        output,
        d_output,
        10 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_input);
}


int GPUMLP::predictClass(
    const float* input
)
{
    float output[10];

    predictGPU(
        input,
        output
    );

    int best = 0;

    for(int i = 1; i < 10; i++)
    {
        if(output[i] > output[best])
        {
            best = i;
        }
    }

    return best;
}
//PRIMER KERNEL
__global__
void computeOutputErrorKernel(
    const float* output,
    const float* target,
    float* errorOutput
)
{
    int k =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(k >= 10)
        return;

    float derivative =
        output[k] > 0.0f ?
        1.0f :
        0.0f;

    errorOutput[k] =
        (target[k] - output[k])
        *
        derivative;
}


__global__
void computeHidden2ErrorKernel(
    const float* weights3,
    const float* errorOutput,
    const float* hidden2,
    float* errorHidden2
)
{
    int j =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(j >= 64)
        return;

    float accumulatedError = 0.0f;

    for(int k = 0; k < 10; k++)
    {
        accumulatedError +=
            weights3[
                k * 64 + j
            ]
            *
            errorOutput[k];
    }

    float derivative =
        hidden2[j] > 0.0f
        ?
        1.0f
        :
        0.0f;

    errorHidden2[j] =
        accumulatedError
        *
        derivative;
}

__global__
void computeHidden1ErrorKernel(
    const float* weights2,
    const float* errorHidden2,
    const float* hidden1,
    float* errorHidden1
)
{
    int j =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(j >= 128)
        return;

    float accumulatedError = 0.0f;

    for(int h = 0; h < 64; h++)
    {
        accumulatedError +=
            weights2[
                h * 128 + j
            ]
            *
            errorHidden2[h];
    }

    float derivative =
        hidden1[j] > 0.0f
        ?
        1.0f
        :
        0.0f;

    errorHidden1[j] =
        accumulatedError
        *
        derivative;
}


__global__
void updateLayer3WeightsKernel(
    float* weights,
    float* biases,
    const float* errorOutput,
    const float* hidden2,
    float learningRate
)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(idx >= 640)
        return;

    int k = idx / 64;
    int j = idx % 64;

    weights[idx] +=
        learningRate *
        errorOutput[k] *
        hidden2[j];

    if(j == 0)
    {
        biases[k] +=
            learningRate *
            errorOutput[k];
    }
}

__global__
void updateLayer2WeightsKernel(
    float* weights,
    float* biases,
    const float* errorHidden2,
    const float* hidden1,
    float learningRate
)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(idx >= 8192)
        return;

    int neuron =
        idx / 128;

    int input =
        idx % 128;

    weights[idx] +=
        learningRate *
        errorHidden2[neuron] *
        hidden1[input];

    if(input == 0)
    {
        biases[neuron] +=
            learningRate *
            errorHidden2[neuron];
    }
}


__global__
void updateLayer1WeightsKernel(
    float* weights,
    float* biases,
    const float* errorHidden1,
    const float* input,
    float learningRate
)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(idx >= 100352)
        return;

    int neuron =
        idx / 784;

    int inputIdx =
        idx % 784;

    weights[idx] +=
        learningRate *
        errorHidden1[neuron] *
        input[inputIdx];

    if(inputIdx == 0)
    {
        biases[neuron] +=
            learningRate *
            errorHidden1[neuron];
    }
}
















__global__
void buildTargetKernel(
    float* target,
    int label
)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(idx >= 10)
        return;

    target[idx] =
        (idx == label)
        ?
        1.0f
        :
        0.0f;
}









GPUMLP::~GPUMLP()
{
    if(d_hidden1)
        cudaFree(d_hidden1);

    if(d_hidden2)
        cudaFree(d_hidden2);

    if(d_output)
        cudaFree(d_output);
    if(d_input)
        cudaFree(d_input);

    if(d_target)
        cudaFree(d_target);

    if(d_errorOutput)
        cudaFree(d_errorOutput);

    if(d_errorHidden2)
        cudaFree(d_errorHidden2);

    if(d_errorHidden1)
        cudaFree(d_errorHidden1);
}

void GPUMLP::trainGPU(
    const float* input,
    int label,
    float learningRate
)
{
    cudaMemcpy(
        d_input,
        input,
        784 * sizeof(float),
        cudaMemcpyHostToDevice
    );

    forwardGPU(
        d_input
    );

    buildTargetKernel<<<1,10>>>(
        d_target,
        label
    );

    cudaDeviceSynchronize();

    computeOutputErrorKernel<<<1,10>>>(
        d_output,
        d_target,
        d_errorOutput
    );

    cudaDeviceSynchronize();

    computeHidden2ErrorKernel<<<1,64>>>(
        layer3.d_weights,
        d_errorOutput,
        d_hidden2,
        d_errorHidden2
    );

    cudaDeviceSynchronize();

    computeHidden1ErrorKernel<<<1,128>>>(
    layer2.d_weights,
    d_errorHidden2,
    d_hidden1,
    d_errorHidden1
);

    cudaDeviceSynchronize();

    updateLayer3WeightsKernel<<<3,256>>>(
        layer3.d_weights,
        layer3.d_biases,
        d_errorOutput,
        d_hidden2,
        learningRate
    );

    cudaDeviceSynchronize();

    updateLayer2WeightsKernel<<<32,256>>>(
        layer2.d_weights,
        layer2.d_biases,
        d_errorHidden2,
        d_hidden1,
        learningRate
    );

    updateLayer1WeightsKernel<<<392,256>>>(
    layer1.d_weights,
    layer1.d_biases,
    d_errorHidden1,
    d_input,
    learningRate
);

    cudaDeviceSynchronize();


    cudaDeviceSynchronize();
}

