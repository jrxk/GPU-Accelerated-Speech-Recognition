#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"
#include "RNN.h"

int main() {
    double startTime = CycleTimer::currentSeconds();
    std::cout << "Hello" << std::endl;

    //Test Linear
    // cuMatrix<float>* x = new cuMatrix<float>(5, 10, 1);
    // // cuMatrix<float> y(10, 10, 1);
    // // cuMatrix<float> z(10, 10, 1);
    // x->toGpu();
    // // y.toGpu();
    // // z.toGpu();
    // Linear* fc = new Linear(5, 10, 20);
    // cuMatrix<float>* y = fc->forward(x);

    //Test RNN (b,i,h,t,l)
    int batch_size = 5;
    int num_layers = 1;
    int time_step = 3;
    int input_size = 10;
    int hidden_size = 20;
    RNN* rnn = new RNN(batch_size, input_size, hidden_size, time_step, num_layers);
    cuMatrix<float> *inputs[batch_size];
    cuMatrix<float> *pre_hiddens[num_layers];
    for (int i = 0; i < time_step; i++) {
        cuMatrix<float>* inp = new cuMatrix<float>(batch_size, input_size, 1);
        inp->toGpu();
        inputs[i] = inp;
    }

    for (int i = 0 ; i < num_layers; i++){
        cuMatrix<float>* hidden = new cuMatrix<float>(batch_size, hidden_size, 1);
        hidden->toGpu();
        pre_hiddens[i] = hidden;
    }

    rnn->forward(inputs, pre_hiddens);



    // matrixMul(&x, &y, &z);
    double endTime = CycleTimer::currentSeconds();
    std::cout << (endTime - startTime) << "s" << std::endl;
}