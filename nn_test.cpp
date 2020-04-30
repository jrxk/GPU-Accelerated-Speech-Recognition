#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"
#include "RNN.h"

int main() {
    // Test Linear
    std::cout << "--- Testing Linear ---" << std::endl;
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
    std::cout << "--- Testing RNN ---" << std::endl;
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
}