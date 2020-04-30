#include "cuMatrix.h"

class Linear
{
public:
    Linear(cuMatrix<float>* weight, cuMatrix<float>* bias, int batch_size, int input_size, int output_size): w(weight), b(bias), batch_size(batch_size), input_size(input_size), output_size(output_size) {
        outputs = new cuMatrix<float>(batch_size, output_size, 1);
        outputs->toGpu();
    }

    Linear(int batch_size, int input_size, int output_size): batch_size(batch_size), input_size(input_size), output_size(output_size) {
        initRandom();
        outputs = new cuMatrix<float>(batch_size, output_size, 1);
        outputs->toGpu();
    }

    void initRandom();
    void initParams(float* w, float* b);
    cuMatrix<float>* forward(cuMatrix<float>* inputs);

    cuMatrix<float>* w;
    cuMatrix<float>* b;
    cuMatrix<float>* outputs;

    int input_size;
    int output_size;
    int batch_size;
};