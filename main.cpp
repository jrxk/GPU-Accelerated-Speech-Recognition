#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"
#include "RNN.h"
#include "CTCBeamSearch.h"

using namespace std;

int main() {

    // Test Linear
    cout << "--- Testing Linear ---" << endl;
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

    // Test RNN
    cout << "--- Testing RNN ---" << endl;
    // seq_len, batch, input_size
    float inp_rnn[4*2*3] = {
        0.1321, 0.0296, 0.2351, 
        0.9742, 0.7064, 0.3638, 
        0.8129, 0.8474, 0.7844, 
        0.9279, 0.9768, 0.7575, 
        0.5693, 0.9383, 0.6537, 
        0.1245, 0.9113, 0.5213, 
        0.2325, 0.2616, 0.2558, 
        0.0063, 0.3980, 0.8896
    };
    float rnn_weight_ih[3*5] = {
        0.0269, -0.1896, 0.0500, 0.1968, -0.2331, 
        -0.1524, -0.1069, -0.3821, 0.3744, -0.0753,
        -0.0177, 0.1578, -0.1543, 0.0330, 0.2318
    };
    float rnn_weight_hh[5*5] = {
        0.0964,  0.3816,  0.1670,  0.2344, -0.0322, 
        -0.3150, 0.2676,  0.1690, 0.1398,  0.0135,
        -0.4383, -0.1151,  0.0135,  0.2061, -0.0159, 
        0.2352, -0.3320, -0.2943,  0.0488, -0.0794, 
        0.2098, -0.0613,  0.3000,  0.2912, -0.0485
    };
    float rnn_bias_ih[5] = {-0.1762,  0.1190,  0.3201, -0.2779, -0.0340};
    float rnn_bias_hh[5] = {-0.1449, -0.0929,  0.0448, -0.0617,  0.4359};

    cuMatrix<float>* inp_test_rnn = new cuMatrix<float>(inp_rnn, 4*2, 3, 1);
    inp_test_rnn->toGpu();
    RNN* rnn_test = new RNN(2, 3, 5, 4, 1);
    rnn_test->rnn_cell[0]->initParams(rnn_weight_ih, rnn_weight_hh, rnn_bias_ih, rnn_bias_hh);

    cuMatrix<float>* out_test_rnn = rnn_test->forward(inp_test_rnn);
    out_test_rnn->toCpu();
    
    // [-0.3151,  0.0350,  0.3130, -0.2865,  0.3998],
    // [-0.3876, -0.1749,  0.0873,  0.1279,  0.2031],
    // [-0.5402, -0.1695,  0.1219,  0.2557,  0.3270],
    // [-0.3853, -0.3751, -0.1476,  0.1991,  0.2695],
    // [-0.3659, -0.4214, -0.1590,  0.1271,  0.3159],
    // [-0.2134, -0.3147, -0.1635, -0.0416,  0.3850],
    // [-0.0956, -0.2925,  0.1586, -0.2606,  0.3544],
    // [-0.1743, -0.0339,  0.1121, -0.1758,  0.5128]
    std::cout << "Output RNN" << std::endl;
    printMatrixInfo(out_test_rnn);

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

    // test CTC
    string v[] = {"","a", "b", "c"};
    int vocabsize = 4;
    float test[] = {0.36225085, 0.09518672, 0.08850375, 0.45405867,
                    0.08869431, 0.18445025, 0.3304224,  0.39643304,
                    0.09951598, 0.17646984, 0.42063249, 0.30338169,
                    0.15361776, 0.46521112, 0.18132693, 0.19984419,
                    0.33478711, 0.16607367, 0.29571415, 0.20342507,
                    0.01292992, 0.36438928, 0.00184853, 0.62083227,
                    0.34142441, 0.16742833, 0.38500542, 0.10614183,
                    0.4443139,  0.12738693, 0.36856127, 0.0597379,
                    0.37673064, 0.13478024, 0.2735787,  0.21491042,
                    0.34790623, 0.04654182, 0.34069546, 0.26485648}; 
    vector<string> vocab (v, v + sizeof(v) / sizeof(string) );
    CTCBeamSearch* decoder = new CTCBeamSearch(vocab, 2, 0);
    cuMatrix<float>* seqProb = new cuMatrix<float>(10, 4, 1);
    for(int j = 0; j < seqProb->getLen(); j++){
        seqProb->getHost()[j] =  test[j];
    }

    seqProb->toGpu();
    string result = decoder->decode(seqProb);
    std::cout << "decoding results: " << result << std::endl;

    // // matrixMul(&x, &y, &z);
    double endTime = CycleTimer::currentSeconds();
    std::cout << (endTime - startTime) << "s" << std::endl;
}