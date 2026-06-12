#ifndef PERCEPTRON_IMAGENES_MNIST_LOADER_H
#define PERCEPTRON_IMAGENES_MNIST_LOADER_H

#pragma once

#include <vector>
#include <string>

using namespace std;

struct MNIST_Data{
    vector<vector<float>> images;
    vector<int> labels;
};

MNIST_Data loadMNIST( const string& imagesPath, const string& labelsPath, int limit);

#endif //PERCEPTRON_IMAGENES_MNIST_LOADER_H