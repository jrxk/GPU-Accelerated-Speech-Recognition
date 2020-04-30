#include "RNN.h"


/*
* inputs: seq * [batch_size, input_size]
* pre_hidden: num_layers * [b, hidden_size]
* output: [b, hidden_size], we need return all hidden outputs for CTC decoder
*/
cuMatrix<float>* RNN::forward(cuMatrix<float>* inputs) {
    cuMatrix<float>* x_t;
    cuMatrix<float>* h_t;
    cuMatrix<float>* h_prev;
    cuMatrix<float>* h_ts[num_layers];

    for (int t = 0; t < time_step; t++){
        // x_t = inputs[t];
        x_t = new cuMatrix<float>(inputs, t * batch_size * input_size, batch_size, input_size, 1);
        for(int l = 0; l < num_layers; l++){
            h_prev = t == 0 ? h_0s[l] : h_ts[l]; 
            h_t = new cuMatrix<float>(hiddens[l], t * batch_size * hidden_size, batch_size, hidden_size, 1);
            rnn_cell[l]->forward(x_t, h_prev, h_t);
            if (t > 0) delete h_prev; // delete temporary matrix
            h_ts[l] = h_t;
            if (l == 0) delete x_t; // delete temporary matrix
            x_t = h_t;
        }
    }

    return hiddens[num_layers-1];
}