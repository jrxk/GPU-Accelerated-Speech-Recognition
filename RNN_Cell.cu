#include "RNN_Cell.h"
#include <cstdlib>

__global__ void Tanh(float* data, float* bias1, float* bias2, int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > height * width) return;
    
    int j = i % width;
    data[i] += bias1[j] + bias2[j];

    data[i] = tanf(data[i]); // tanf(data[i] * 2.0 / 3.0) * 1.7159 or exp-exp/exp+exp LOL
}

void RNN_Cell::initRandom() {
    this->w_ih= new cuMatrix<float>(input_size, hidden_size, 1);
    this->w_hh= new cuMatrix<float>(hidden_size, hidden_size, 1);
    this->b_ih = new cuMatrix<float>(hidden_size, 1, 1);
    this->b_hh = new cuMatrix<float>(hidden_size, 1, 1);

    for(int i = 0; i < w_ih->getLen(); i++){
        w_ih->getHost()[i] =  (2.0f * rand() / RAND_MAX - 1.0f);
    }

    for(int j = 0; j < w_hh->getLen(); j++){
        w_hh->getHost()[j] =  (2.0f * rand() / RAND_MAX - 1.0f);
    }

    this->w_ih->toGpu();
    this->w_hh->toGpu();
    this->b_ih->toGpu();
    this->b_hh->toGpu();
}

/*
* input: [b, input_size], here b is real batch 
* pre_hidden: [b, hidden_size]
* output: [b, hidden_size]
*/
cuMatrix<float>* RNN_Cell::forward(cuMatrix<float>* inputs, cuMatrix<float>* pre_hidden, cuMatrix<float>* outputs) {
    matrixMul(inputs, w_ih, ih_outputs);
    matrixMul(pre_hidden, w_hh, hh_outputs);
    matrixAdd(ih_outputs, hh_outputs, outputs, 1);

    int blockDim = 256;
    int numBlocks = (batch_size * hidden_size + blockDim - 1) / blockDim;
    Tanh<<<numBlocks, blockDim>>>(outputs->getDev(), b_hh->getDev(), b_ih->getDev(), batch_size, hidden_size);
    return outputs;
}