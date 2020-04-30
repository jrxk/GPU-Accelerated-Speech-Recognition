#include "Linear.h"

__global__ void ReLU(float* data, float* bias, int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > height * width) return;
    
    int j = i % width;
    data[i] += bias[j];
    if (data[i] < 0) data[i] = 0; // fmaxf ?
}

void Linear::initRandom() {
    this->w = new cuMatrix<float>(input_size, output_size, 1);
    this->b = new cuMatrix<float>(output_size, 1, 1);

    for(int j = 0; j < w->getLen(); j++){
        w->getHost()[j] =  (2.0f * rand() / RAND_MAX - 1.0f);
    }
    this->w->toGpu();
    this->b->toGpu();
}

void Linear::initParams(float* weight, float* bias) {
    this->w = new cuMatrix<float>(input_size, output_size, 1);
    this->b = new cuMatrix<float>(output_size, 1, 1);

    for (int j = 0; j < w->getLen(); j++){
        w->getHost()[j] =  weight[j];
    }
    for (int j = 0; j < b->getLen(); j++){
        b->getHost()[j] =  bias[j];
    }

    this->w->toGpu();
    this->b->toGpu();
}

/*
* input: [b, input_size], b is batch_size x seq_len 
* output: [b, output_size]
*/
cuMatrix<float>* Linear::forward(cuMatrix<float>* inputs) {
    matrixMul(inputs, w, outputs);
    outputs->toCpu();
    int blockDim = 256;
    int numBlocks = (batch_size * output_size + blockDim - 1) / blockDim;
    ReLU<<<numBlocks, blockDim>>>(outputs->getDev(), b->getDev(), batch_size, output_size);
    return outputs;
}