#include "cuMatrix.h"
#include "RNN_Cell.h"

class RNN
{
public:

    RNN(int batch_size, int input_size, int hidden_size, int time_step, int num_layers): batch_size(batch_size), input_size(input_size), hidden_size(hidden_size), time_step(time_step), num_layers(num_layers) {
        // outputs = new cuMatrix<float>(batch_size, time_step * hidden_size, 1);
        rnn_cell = new RNN_Cell*[num_layers];
        h_0s = new cuMatrix<float>*[num_layers];
        hiddens = new cuMatrix<float>*[num_layers];
        for (int i = 0; i < num_layers; i++){
            int _input_size = i == 0 ? input_size : hidden_size;
            rnn_cell[i] = new RNN_Cell(batch_size, _input_size, hidden_size);
            h_0s[i] = new cuMatrix<float>(batch_size, hidden_size, 1);
            h_0s[i]->toGpu();
            hiddens[i] = new cuMatrix<float>(time_step * batch_size, hidden_size, 1);
            hiddens[i]->toGpu();
        }
    }

    cuMatrix<float>* forward(cuMatrix<float>* inputs);

    cuMatrix<float>** h_0s; // num_layers * [batch_size, hidden_size]
    cuMatrix<float>** hiddens; // num_layers * [seq_len * batch, hidden_size]
    // cuMatrix<float>** outputs; // time_step * [batch_size, hidden_size]
    // cuMatrix<float>** hidden_layer_outputs; // num_layers * batch * hidden_size

    RNN_Cell** rnn_cell;
    int input_size;
    int hidden_size;
    int batch_size;
    int time_step;
    int num_layers; 

};