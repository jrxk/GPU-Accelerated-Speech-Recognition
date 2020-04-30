#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"
#include "RNN.h"

int main() {

    // Test Linear
    float inp[6] = {0.0932, 0.3362, 0.1910, 0.6148, 0.5331, 0.1238};
    float weight[12] = {0.5699999928474426, 0.03020000085234642, -0.22759999334812164, 0.1242000013589859, 0.34470000863075256, 0.49300000071525574, 0.37700000405311584, 0.04749999940395355, 0.3377000093460083, -0.4636000096797943, -0.5188999772071838, 0.09910000115633011};
    float bias[4] = {0.37158000469207764, -0.4036799967288971, 0.21911999583244324, 0.0001550900051370263};
    cuMatrix<float>* inp_test = new cuMatrix<float>(inp, 2, 3, 1);
    inp_test->toGpu();
    Linear* mlp_test = new Linear(2, 3, 4);
    mlp_test->initParams(weight, bias);
    cuMatrix<float>* out_test = mlp_test->forward(inp_test);
    out_test->toCpu();
    
    std::cout << "Input" << std::endl;
    printMatrixInfo(inp_test);
    
    std::cout << "Weight" << std::endl;
    printMatrixInfo(mlp_test->w);

    std::cout << "Bias" << std::endl;
    printMatrixInfo(mlp_test->b);

    // 0.6051, 0.0000, 0.2255, 0.0466,
    // 0.9476, 0.0000, 0.2159, 0.1141
    std::cout << "Output" << std::endl;
    printMatrixInfo(out_test);

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

    Linear* mlp1 = new Linear(batch_size * seq_len, input_size, hidden_1);
    Linear* mlp2 = new Linear(batch_size * seq_len, hidden_1, hidden_2);
    Linear* mlp3 = new Linear(batch_size * seq_len, hidden_2, hidden_3);
    RNN* rnn = new RNN(batch_size, rnn_hidden_size, rnn_hidden_size, seq_len, rnn_num_layers);
    Linear* mlp5 = new Linear(batch_size * seq_len, rnn_hidden_size, hidden_5);
    Linear* mlp6 = new Linear(batch_size * seq_len, hidden_5, output_size);

    cuMatrix<float>* x = new cuMatrix<float>(batch_size * seq_len, input_size, 1);
    x->toGpu();
    cuMatrix<float>* x1 = mlp1->forward(x);
    cuMatrix<float>* x2 = mlp2->forward(x1);
    cuMatrix<float>* x3 = mlp3->forward(x2);
    cuMatrix<float>* x4 = rnn->forward(x3);
    cuMatrix<float>* x5 = mlp5->forward(x4);
    cuMatrix<float>* out = mlp6->forward(x5);

    // Initialize RNN inputs
    // cuMatrix<float>** x_seq = new cuMatrix<float>*[seq_len];
    // for (int i = 0; i < seq_len; i++) { 
    //     x_seq[i] = new cuMatrix<float>(x3, i * batch_size * hidden_3, batch_size, hidden_3, 1);
    //     x_seq[i]->set(0, 0, 0, float(i));
    // }
    // Verify slicing is correct
    // for (int i = 0; i < seq_len; i++) { 
    //     std::cout << x_seq[i]->get(0, 0, 0) << std::endl;
    //     std::cout << x3->get(i * batch_size, 0, 0) << std::endl;
    // }

    // cuMatrix<float>** init_hiddens = new cuMatrix<float>*[rnn_num_layers];
    // for (int i = 0 ; i < rnn_num_layers; i++){
    //     cuMatrix<float>* hidden = new cuMatrix<float>(batch_size, rnn_hidden_size, 1);
    //     hidden->toGpu();
    //     init_hiddens[i] = hidden;
    // }





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


    // cuMatrix<float>* x = new cuMatrix<float>(5, 10, 1);
    // x->toGpu();
    // y.toGpu();
    // z.toGpu();
    // Linear* fc = new Linear(5, 10, 20);
    // cuMatrix<float>* y = fc->forward(x);

    // // matrixMul(&x, &y, &z);
    double endTime = CycleTimer::currentSeconds();
    std::cout << (endTime - startTime) << "s" << std::endl;
}