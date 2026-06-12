#include <iostream>

#include "gpu_mlp.h"
#include "../data/mnist_loader.h"

using namespace std;

int main()
{
    GPUMLP model;

    model.copyToGPU();

    cout << "Cargando entrenamiento...\n";

    MNIST_Data train =
        loadMNIST(
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            60000
        );

    cout << "\nCargando test...\n";

    MNIST_Data test =
        loadMNIST(
            "../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte",
            10000
        );

    int epochs = 15;

    cout << "\nIniciando entrenamiento...\n";

    for(int epoch = 0; epoch < epochs; epoch++)
    {
        for(size_t i = 0;
            i < train.images.size();
            i++)
        {
            model.trainGPU(
                train.images[i].data(),
                train.labels[i],
                0.01f
            );
        }

        int correct = 0;

        for(size_t i = 0;
            i < test.images.size();
            i++)
        {
            int prediction =
                model.predictClass(
                    test.images[i].data()
                );

            if(prediction == test.labels[i])
            {
                correct++;
            }
        }

        cout
            << "Epoch "
            << (epoch + 1)
            << " Accuracy: "
            << (100.0f * correct / test.images.size())
            << "%"
            << endl;
    }

    return 0;
}