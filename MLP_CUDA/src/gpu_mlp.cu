#include "gpu_mlp.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

__global__
void denseForwardKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int inputSize,
    int outputSize,
    int batchSize
)
{
    int neuron =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int sample =
        blockIdx.y;

    if(neuron >= outputSize)
        return;

    if(sample >= batchSize)
        return;

    float sum =
        biases[neuron];

    const float* sampleInput =
        input +
        sample * inputSize;

    for(int i = 0; i < inputSize; i++)
    {
        sum +=
            weights[
                neuron * inputSize + i
            ]
            *
            sampleInput[i];
    }

    output[
        sample * outputSize +
        neuron
    ] =
        sum > 0.0f
        ?
        sum
        :
        0.0f;
}

__global__
void denseOutputKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int inputSize,
    int outputSize,
    int batchSize
)
{
    int neuron =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int sample =
        blockIdx.y;

    if(neuron >= outputSize)
        return;

    if(sample >= batchSize)
        return;

    float sum =
        biases[neuron];

    const float* sampleInput =
        input +
        sample * inputSize;

    for(int i = 0;
        i < inputSize;
        i++)
    {
        sum +=
            weights[
                neuron * inputSize + i
            ]
            *
            sampleInput[i];
    }

    output[
        sample * outputSize +
        neuron
    ] = sum;
}



__global__
void softmaxKernel(
    float* output,
    int batchSize
)
{
    int sample = blockIdx.x;

    if(sample >= batchSize)
        return;

    float* row =
        output + sample * 10;

    float maxValue = row[0];

    for(int i = 1; i < 10; i++)
    {
        if(row[i] > maxValue)
            maxValue = row[i];
    }

    float sumExp = 0.0f;

    for(int i = 0; i < 10; i++)
    {
        row[i] = expf(
            row[i] - maxValue
        );

        sumExp += row[i];
    }

    for(int i = 0; i < 10; i++)
    {
        row[i] /= sumExp;
    }
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
//MINI BATCHES
    cudaMalloc(
        &d_hidden1,
        BATCH_SIZE *
        128 *
        sizeof(float)
    );

    cudaMalloc(
        &d_hidden2,
        BATCH_SIZE *
        64 *
        sizeof(float)
    );


    cudaMalloc(
        &d_output,
        BATCH_SIZE *
        10 *
        sizeof(float)
    );
    //waaaaaaa

    cudaMalloc(
    &d_batchInput,
    BATCH_SIZE *
    784 *
    sizeof(float)
);

    cudaMalloc(
        &d_batchLabels,
        BATCH_SIZE *
        sizeof(int)
    );



    cudaMalloc(
        &d_target,
        BATCH_SIZE *
        10 *
        sizeof(float)
    );

    cudaMalloc(
        &d_errorOutput,
        BATCH_SIZE *
        10 *
        sizeof(float)
    );

    cudaMalloc(
        &d_errorHidden1,
        BATCH_SIZE *
        128 *
        sizeof(float)
    );
    cudaMalloc(
        &d_errorHidden2,
        BATCH_SIZE *
        64 *
        sizeof(float)
    );

}void GPUMLP::uploadDataset(
    const std::vector<float>& images,
    const std::vector<int>& labels
)
{


    cudaMalloc(
        &d_trainImages,
        images.size() * sizeof(float)
    );

    cudaMemcpy(
        d_trainImages,
        images.data(),
        images.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMalloc(
        &d_trainLabels,
        labels.size() * sizeof(int)
    );

    cudaMemcpy(
        d_trainLabels,
        labels.data(),
        labels.size() * sizeof(int),
        cudaMemcpyHostToDevice
    );
}








void GPUMLP::copyToGPU()
{
    layer1.copyToGPU();
    layer2.copyToGPU();
    layer3.copyToGPU();
}

void GPUMLP::forwardGPU(
    const float* batchInput,
    int batchSize
) {
    dim3 block1(128);
    dim3 grid1(1, batchSize);

    denseForwardKernel<<<
        grid1,
        block1
    >>>(
        batchInput,
        layer1.d_weights,
        layer1.d_biases,
        d_hidden1,
        784,
        128,
        batchSize
    );

    dim3 block2(64);
    dim3 grid2(1, batchSize);

    denseForwardKernel<<<
        grid2,
        block2
    >>>(
        d_hidden1,
        layer2.d_weights,
        layer2.d_biases,
        d_hidden2,
        128,
        64,
        batchSize
    );

    dim3 block3(16);

    dim3 grid3(
        (10 + block3.x - 1) /
        block3.x,
        batchSize
    );

    denseOutputKernel<<<
        grid3,
        block3
    >>>(
        d_hidden2,
        layer3.d_weights,
        layer3.d_biases,
        d_output,
        64,
        10,
        batchSize
    );
    softmaxKernel<<<
    batchSize,
    1
>>>(
    d_output,
    batchSize
);

}


void GPUMLP::predictGPU(
    const float* input,
    float* output,
    int batchSize
)
{



    //batch
    cudaMemcpy(
    d_batchInput,
    input,
    batchSize * 784 * sizeof(float),
    cudaMemcpyHostToDevice
);

    forwardGPU(d_batchInput,batchSize);

    cudaMemcpy(
    output,
    d_output,
    batchSize * 10 * sizeof(float),
    cudaMemcpyDeviceToHost
);


}


int GPUMLP::predictClass(
    const float* input
)
{
    float output[10];

    predictGPU(
        input,
        output,
        1
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
void computeOutputErrorBatchKernel(
    const float* output,
    const float* target,
    float* errorOutput,
    int batchSize
)
{
    int classIdx =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(classIdx >= 10)
        return;

    int idx =
        sample * 10 +
        classIdx;

    errorOutput[idx] =
        output[idx]
        -
        target[idx];
}
__global__
void computeHidden2ErrorBatchKernel(
    const float* errorOutput,
    const float* hidden2,
    const float* weights3,
    float* errorHidden2,
    int batchSize
)
{
    int neuron =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(neuron >= 64)
        return;

    float accumulatedError =
        0.0f;

    for(int k = 0; k < 10; k++)
    {
        accumulatedError +=
            weights3[
                k * 64 +
                neuron
            ]
            *
            errorOutput[
                sample * 10 +
                k
            ];
    }

    float derivative =
        hidden2[
            sample * 64 +
            neuron
        ] > 0.0f
        ?
        1.0f
        :
        0.0f;

    errorHidden2[
        sample * 64 +
        neuron
    ]
    =
    accumulatedError
    *
    derivative;
}
__global__
void computeHidden1ErrorBatchKernel(
    const float* errorHidden2,
    const float* hidden1,
    const float* weights2,
    float* errorHidden1,
    int batchSize
)
{
    int neuron =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(neuron >= 128)
        return;

    float accumulatedError =
        0.0f;

    for(int h = 0; h < 64; h++)
    {
        accumulatedError +=
            weights2[
                h * 128 +
                neuron
            ]
            *
            errorHidden2[
                sample * 64 +
                h
            ];
    }

    float derivative =
        hidden1[
            sample * 128 +
            neuron
        ] > 0.0f
        ?
        1.0f
        :
        0.0f;

    errorHidden1[
        sample * 128 +
        neuron
    ]
    =
    accumulatedError
    *
    derivative;
}


__global__
void updateLayer3WeightsBatchKernel(
    float* weights,
    const float* hidden2,
    const float* errorOutput,
    float learningRate,
    int batchSize
)
{
    int inputIdx =
    blockIdx.x * blockDim.x +
    threadIdx.x;

    int neuron =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if(
    neuron >= 10 ||
    inputIdx >= 64
)
        return;

    float gradient =
        0.0f;

    for(int sample = 0;
        sample < batchSize;
        sample++)
    {
        gradient +=
            errorOutput[
                sample * 10 +
                neuron
            ]
            *
            hidden2[
                sample * 64 +
                inputIdx
            ];
    }

    gradient /= batchSize;

    weights[
        neuron * 64 +
        inputIdx
    ]
    -=
    learningRate
    *
    gradient;


}


__global__
void updateLayer3BiasesKernel(
    float* biases,
    const float* errorOutput,
    float learningRate,
    int batchSize
)
{
    int neuron =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if(neuron >= 10)
        return;

    float gradient = 0.0f;

    for(int sample = 0;
        sample < batchSize;
        sample++)
    {
        gradient +=
            errorOutput[
                sample * 10 +
                neuron
            ];
    }

    gradient /= batchSize;

    biases[neuron] -=
        learningRate *
        gradient;
}

__global__
void updateLayer2WeightsBatchKernel(
    float* weights,
    const float* hidden1,
    const float* errorHidden2,
    float learningRate,
    int batchSize
)
{
    int inputIdx =
    blockIdx.x * blockDim.x +
    threadIdx.x;

    int neuron =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if(
    neuron >= 64 ||
    inputIdx >= 128
)
        return;

    float gradient = 0.0f;

    for(int sample = 0;
        sample < batchSize;
        sample++)
    {
        gradient +=
            errorHidden2[
                sample * 64 +
                neuron
            ]
            *
            hidden1[
                sample * 128 +
                inputIdx
            ];
    }

    gradient /= batchSize;

    weights[
        neuron * 128 +
        inputIdx
    ]
    -=
    learningRate *
    gradient;

}

__global__
void updateLayer2BiasesKernel(
    float* biases,
    const float* errorHidden2,
    float learningRate,
    int batchSize
)
{
    int neuron =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if(neuron >= 64)
        return;

    float gradient = 0.0f;

    for(int sample = 0;
        sample < batchSize;
        sample++)
    {
        gradient +=
            errorHidden2[
                sample * 64 +
                neuron
            ];
    }

    gradient /= batchSize;

    biases[neuron] -=
        learningRate *
        gradient;
}
__global__
void updateLayer1WeightsBatchKernel(
    float* weights,
    const float* batchInput,
    const float* errorHidden1,
    float learningRate,
    int batchSize
)
{
    int inputIdx =
    blockIdx.x * blockDim.x +
    threadIdx.x;

    int neuron =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if(
    neuron >= 128 ||
    inputIdx >= 784
)
        return;

    float gradient =
        0.0f;

    for(int sample = 0;
        sample < batchSize;
        sample++)
    {
        gradient +=
            errorHidden1[
                sample * 128 +
                neuron
            ]
            *
            batchInput[
                sample * 784 +
                inputIdx
            ];
    }

    gradient /= batchSize;

    weights[
        neuron * 784 +
        inputIdx
    ]
    -=
    learningRate *
    gradient;


}

__global__
void updateLayer1BiasesKernel(
    float* biases,
    const float* errorHidden1,
    float learningRate,
    int batchSize
)
{
    int neuron =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if(neuron >= 128)
        return;

    float gradient = 0.0f;

    for(int sample = 0;
        sample < batchSize;
        sample++)
    {
        gradient +=
            errorHidden1[
                sample * 128 +
                neuron
            ];
    }

    gradient /= batchSize;

    biases[neuron] -=
        learningRate *
        gradient;
}












__global__
void buildTargetBatchKernel(
    float* target,
    const int* labels,
    int batchSize
)
{
    int classIdx =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(classIdx >= 10)
        return;

    target[
        sample * 10 +
        classIdx
    ]
    =
    (
        classIdx ==
        labels[sample]
    )
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
    //mini batches
    if(d_batchInput)
        cudaFree(d_batchInput);

    if(d_batchLabels)
        cudaFree(d_batchLabels);
    //
    if(d_target)
        cudaFree(d_target);

    if(d_errorOutput)
        cudaFree(d_errorOutput);

    if(d_errorHidden2)
        cudaFree(d_errorHidden2);

    if(d_errorHidden1)
        cudaFree(d_errorHidden1);
    if(d_trainImages)
        cudaFree(d_trainImages);

    if(d_trainLabels)
        cudaFree(d_trainLabels);
}


void GPUMLP::trainBatchFromGPU(
    int batchStart,
    int batchSize,
    float learningRate
)
{
    const float* batchImages =
        d_trainImages +
        batchStart * 784;

    const int* batchLabels =
        d_trainLabels +
        batchStart;



    cudaMemcpy(
        d_batchLabels,
        batchLabels,
        batchSize *
        sizeof(int),
        cudaMemcpyDeviceToDevice
    );

    forwardGPU(batchImages,batchSize);

    buildTargetBatchKernel<<<batchSize,10>>>(
    d_target,
    d_batchLabels,
    batchSize
);


    computeOutputErrorBatchKernel<<<
    batchSize,
    10
>>>(
    d_output,
    d_target,
    d_errorOutput,
    batchSize
);



    computeHidden2ErrorBatchKernel<<<
    batchSize,
    64
>>>(
    d_errorOutput,
    d_hidden2,
    layer3.d_weights,
    d_errorHidden2,
    batchSize
);



    computeHidden1ErrorBatchKernel<<<
        batchSize,
        128
    >>>(
        d_errorHidden2,
        d_hidden1,
        layer2.d_weights,
        d_errorHidden1,
        batchSize
    );




    dim3 blockLayer3(16,16);

    dim3 gridLayer3(
        (64 + blockLayer3.x - 1) /
        blockLayer3.x,

        (10 + blockLayer3.y - 1) /
        blockLayer3.y
    );

    updateLayer3WeightsBatchKernel<<<
        gridLayer3,
        blockLayer3
    >>>(
        layer3.d_weights,
        d_hidden2,
        d_errorOutput,
        learningRate,
        batchSize
    );

    updateLayer3BiasesKernel<<<
        1,
        10
    >>>(
        layer3.d_biases,
        d_errorOutput,
        learningRate,
        batchSize
    );




    dim3 blockLayer2(16,16);

    dim3 gridLayer2(
        (128 + blockLayer2.x - 1) /
        blockLayer2.x,

        (64 + blockLayer2.y - 1) /
        blockLayer2.y
    );

    updateLayer2WeightsBatchKernel<<<
        gridLayer2,
        blockLayer2
    >>>(
        layer2.d_weights,
        d_hidden1,
        d_errorHidden2,
        learningRate,
        batchSize
    );
    updateLayer2BiasesKernel<<<
        1,
        64
    >>>(
        layer2.d_biases,
        d_errorHidden2,
        learningRate,
        batchSize
    );


    dim3 block(16,16);

    dim3 grid(
        (784 + block.x - 1) / block.x,
        (128 + block.y - 1) / block.y
    );

    updateLayer1WeightsBatchKernel<<<
        grid,
        block
    >>>(
        layer1.d_weights,
        batchImages,
        d_errorHidden1,
        learningRate,
        batchSize
    );

    updateLayer1BiasesKernel<<<
    1,
    128
>>>(
    layer1.d_biases,
    d_errorHidden1,
    learningRate,
    batchSize
);


    cudaDeviceSynchronize();

}








