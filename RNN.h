#include "cuMatrix.h"
#include "RNN_Cell.h"

class RNN
{
public:

    RNN(int batch_size, int input_size, int hidden_size, int time_step, int num_layers): batch_size(batch_size), input_size(input_size), hidden_size(hidden_size), time_step(time_step), num_layers(num_layers) {
        // outputs = new cuMatrix<float>(batch_size, time_step * hidden_size, 1);
        
        for (int i = 0; i < num_layers; i++){
            rnn_cell[i] = new RNN_Cell(batch_size, input_size, hidden_size);
            hidden_layer_outputs[i] = new cuMatrix<float>(batch_size, hidden_size, 1);
        }

        // outputs->toGpu();
    }

    cuMatrix<float>** forward(cuMatrix<float>** inputs, cuMatrix<float>** pre_hidden);

private:
    
    cuMatrix<float>** outputs; // time_step * [batch_size, hidden_size]
    cuMatrix<float>** hidden_layer_outputs; // num_layers * batch * hidden_size

    RNN_Cell** rnn_cell;
    int input_size;
    int hidden_size;
    int batch_size;
    int time_step;
    int num_layers; 

};