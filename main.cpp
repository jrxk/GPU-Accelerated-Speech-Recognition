#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"
#include "Linear.h"

int main() {
    double startTime = CycleTimer::currentSeconds();
    std::cout << "Hello" << std::endl;
    cuMatrix<float>* x = new cuMatrix<float>(5, 10, 1);
    // cuMatrix<float> y(10, 10, 1);
    // cuMatrix<float> z(10, 10, 1);
    x->toGpu();
    // y.toGpu();
    // z.toGpu();
    Linear* fc = new Linear(5, 10, 20);
    cuMatrix<float>* y = fc->forward(x);

    // matrixMul(&x, &y, &z);
    double endTime = CycleTimer::currentSeconds();
    std::cout << (endTime - startTime) << "s" << std::endl;
}