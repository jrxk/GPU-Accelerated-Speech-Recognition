#include <iostream>
#include "cuMatrix.h"
#include "cycleTimer.h"

int main() {
    double startTime = CycleTimer::currentSeconds();
    std::cout << "Hello" << std::endl;
    cuMatrix<float> x(10, 10, 1);
    cuMatrix<float> y(10, 10, 1);
    cuMatrix<float> z(10, 10, 1);
    x.toGpu();
    y.toGpu();
    z.toGpu();
    matrixMul(&x, &y, &z);
    double endTime = CycleTimer::currentSeconds();
    std::cout << (endTime - startTime) << "s" << std::endl;
}