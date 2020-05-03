#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"
#include "RNN.h"
#include "CTCBeamSearch.h"

#include <thrust/device_vector.h>

using namespace std;

int main() {
    double startTime = CycleTimer::currentSeconds();
    std::cout << "--- Deep Dark Speech ---" << std::endl;
    // int rnn_num_layers = 1;

    // int n_context = 1;
    // int input_size = 10;
    // int linear_size = 40;
    // int rnn_hidden_size = 50;
    // int vocab_size = 26;

    // int hidden_1 = linear_size;
    // int hidden_2 = linear_size;
    // int hidden_5 = linear_size;
    // int hidden_3 = rnn_hidden_size;
    // int output_size = vocab_size + 1;
    // int batch_size = 3;
    // int seq_len = 9;

    // Linear* mlp1 = new Linear(batch_size * seq_len, input_size, hidden_1);
    // Linear* mlp2 = new Linear(batch_size * seq_len, hidden_1, hidden_2);
    // Linear* mlp3 = new Linear(batch_size * seq_len, hidden_2, hidden_3);
    // RNN* rnn = new RNN(batch_size, rnn_hidden_size, rnn_hidden_size, seq_len, rnn_num_layers);
    // Linear* mlp5 = new Linear(batch_size * seq_len, rnn_hidden_size, hidden_5);
    // Linear* mlp6 = new Linear(batch_size * seq_len, hidden_5, output_size);

    // cuMatrix<float>* x = new cuMatrix<float>(batch_size * seq_len, input_size, 1);
    // x->toGpu();
    // cuMatrix<float>* x1 = mlp1->forward(x);
    // cuMatrix<float>* x2 = mlp2->forward(x1);
    // cuMatrix<float>* x3 = mlp3->forward(x2);
    // cuMatrix<float>* x4 = rnn->forward(x3);
    // cuMatrix<float>* x5 = mlp5->forward(x4);
    // cuMatrix<float>* out = mlp6->forward(x5);

    // test CTC
    char vocab[] = {'$','a', 'b', 'c'}; // blank in the vocab must be a special symbol
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
    // vector<string> vocab (v, v + sizeof(v) / sizeof(string) );
    CTCBeamSearch* decoder = new CTCBeamSearch(vocab, vocabsize, 3000, 0);
    cuMatrix<float>* seqProb = new cuMatrix<float>(10, 4, 1);
    for(int j = 0; j < seqProb->getLen(); j++){
        seqProb->getHost()[j] =  test[j];
    }

    seqProb->toGpu();
    string result = decoder->decode(seqProb, 10, 1);
    std::cout << "decoding results: " << result << std::endl;

    // // matrixMul(&x, &y, &z);
    double endTime = CycleTimer::currentSeconds();
    std::cout << (endTime - startTime) << "s" << std::endl;
}