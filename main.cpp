#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"
#include "RNN.h"

int main() {
    double startTime = CycleTimer::currentSeconds();
    std::cout << "--- Deep Dark Speech ---" << std::endl;
    int rnn_num_layers = 1;

    int n_context = 1;
    int input_size = 10;
    int linear_size = 40;
    int rnn_hidden_size = 50;
    int vocab_size = 26;

    int hidden_1 = linear_size;
    int hidden_2 = linear_size;
    int hidden_5 = linear_size;
    int hidden_3 = rnn_hidden_size;
    int output_size = vocab_size + 1;
    int batch_size = 3;
    int seq_len = 9;
    
    //Test Linear
    // cuMatrix<float>* x = new cuMatrix<float>(5, 10, 1);
    // x->toGpu();
    // // y.toGpu();
    // // z.toGpu();
    // Linear* fc = new Linear(5, 10, 20);
    // cuMatrix<float>* y = fc->forward(x);

    Linear* mlp1 = new Linear(batch_size*seq_len, input_size, hidden_1);
    Linear* mlp2 = new Linear(batch_size*seq_len, hidden_1, hidden_2);
    Linear* mlp3 = new Linear(batch_size*seq_len, hidden_2, hidden_3);
    // RNN* rnn = new RNN(batch_size, input_size, rnn_hidden_size, seq_len, rnn_num_layers);

    cuMatrix<float>* x = new cuMatrix<float>(batch_size * seq_len, input_size, 1);
    x->toGpu();
    cuMatrix<float>* x1 = mlp1->forward(x);
    cuMatrix<float>* x2 = mlp2->forward(x1);
    cuMatrix<float>* x3 = mlp3->forward(x2);


    // //Test RNN (b,i,h,t,l)
    // int batch_size = 5;
    // int num_layers = 1;
    // int time_step = 3;
    // int input_size = 10;
    // int hidden_size = 20;
    // RNN* rnn = new RNN(batch_size, input_size, hidden_size, time_step, num_layers);
    // cuMatrix<float> *inputs[batch_size];
    // cuMatrix<float> *pre_hiddens[num_layers];
    // for (int i = 0; i < time_step; i++) {
    //     cuMatrix<float>* inp = new cuMatrix<float>(batch_size, input_size, 1);
    //     inp->toGpu();
    //     inputs[i] = inp;
    // }

    // for (int i = 0 ; i < num_layers; i++){
    //     cuMatrix<float>* hidden = new cuMatrix<float>(batch_size, hidden_size, 1);
    //     hidden->toGpu();
    //     pre_hiddens[i] = hidden;
    // }

    // rnn->forward(inputs, pre_hiddens);



    // // matrixMul(&x, &y, &z);
    // double endTime = CycleTimer::currentSeconds();
    // std::cout << (endTime - startTime) << "s" << std::endl;
}