#include <iostream>
#include <algorithm>
#include "gpu_mlp.cuh"
#include "../data/mnist_loader.h"

using namespace std;

int main()
{
    GPUMLP model;

    const int BATCH_SIZE = 64;
    const int EPOCHS = 128;

    model.copyToGPU();

    cout << "Cargando entrenamiento...\n";

    MNIST_Data train =
        loadMNIST(
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            60000
        );

    model.uploadDataset(train.images,train.labels);

    cout << "\nCargando test...\n";

    MNIST_Data test =
        loadMNIST(
            "../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte",
            10000
        );

    cout << "\nIniciando entrenamiento...\n";

    for(int epoch=0; epoch<EPOCHS; epoch++)
    {
        for(
            int batchStart=0;
            batchStart<train.imageCount;
            batchStart+=BATCH_SIZE
        )
        {
            int currentBatchSize =
                min(
                    BATCH_SIZE,
                    train.imageCount - batchStart
                );

            model.trainBatchFromGPU(
                batchStart,
                currentBatchSize,
                0.01f
            );
        }

        cout
            << "Epoch "
            << epoch + 1
            << " completada\n";
    }

    cout << "\nProbando prediccion...\n";

    int correct = 0;

    for(int i = 0;
        i < test.imageCount;
        i++)
    {
        const float* image =
            &test.images[
                i * test.imageSize
            ];

        int pred =
            model.predictClass(image);

        if(pred == test.labels[i])
            correct++;
    }

    float accuracy =
    100.0f *
    static_cast<float>(correct)
    /
    test.imageCount;

    cout
        << "\nTest Accuracy: "
        << accuracy
        << "%"
        << endl;

    return 0;
}