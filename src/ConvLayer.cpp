#include "ConvLayer.h"

ConvLayer::ConvLayer(int prevMatLength, int kernelLength) {
        initWeights(prevMatLength,kernelLength);
}

void ConvLayer::initWeights(int prevMatLength, int kernelLength) {
    kernels = Eigen::MatrixXf::Random(kernelLength,kernelLength);
    biases = Eigen::MatrixXf::Random(1+prevMatLength-kernelLength);
}

void ConvLayer::forwardProp(Eigen::MatrixXf X) {}

void ConvLayer::backProp() {}
    
void ConvLayer::gradDesc() {}

Eigen::MatrixXf ConvLayer::getA() {}
