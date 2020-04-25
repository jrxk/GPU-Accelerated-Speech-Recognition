#include "cuMatrix.h"

class RNN_Cell
{
public:
    RNN_Cell(int batch_size, int input_size, int hidden_size): batch_size(batch_size), input_size(input_size), hidden_size(hidden_size) {
        initRandom();
        outputs = new cuMatrix<float>(batch_size, hidden_size, 1);
        hh_outputs = new cuMatrix<float>(batch_size, hidden_size, 1);
        ih_outputs = new cuMatrix<float>(batch_size, hidden_size, 1);
        outputs->toGpu();
        hh_outputs->toGpu();
        ih_outputs->toGpu();
    }

    void initRandom();
    cuMatrix<float>* forward(cuMatrix<float>* inputs, cuMatrix<float>* pre_hidden);

private:
    cuMatrix<float>* w_ih; // input_size * hidden_size
    cuMatrix<float>* w_hh; // hidden_size * hidden_size
    cuMatrix<float>* b_ih; // hidden_size
    cuMatrix<float>* b_hh; // hidden_size
    // Or can we only use 1 bias (same bias for 2 parts)?
    // Or We can concat input and pre_hidden, just using one weight?

    cuMatrix<float>* outputs; // batch_size * hidden_size
    
    //Are these 2 necessary?
    cuMatrix<float>* hh_outputs;
    cuMatrix<float>* ih_outputs;

    int input_size;
    int hidden_size;
    int batch_size;

};