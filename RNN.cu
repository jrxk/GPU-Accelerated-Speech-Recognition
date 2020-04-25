#include "RNN.h"


/*
* inputs: [seq, (batch_size * input_size)]
* pre_hidden: num_layers * [b, hidden_size]
* output: [b, hidden_size], we need return all hidden outputs for CTC decoder
*/
cuMatrix<float>* RNN::forward(cuMatrix<float>* inputs, cuMatrix<float>** pre_hidden) {
    cuMatrix<float>* current_input;
    
    for (int t = 0; t < time_step; t++){
        current_input = inputs[t];  // ？
        for(int l = 0; l < num_layers; l++){
            pre_hidden[l] = rnn_cell[l]->forward(current_input, pre_hidden[l]);
            current_input = pre_hidden[l];
        }

        outputs[t] = pre_hidden[num_layers - 1]->getDev();
    }

    return outputs;
}