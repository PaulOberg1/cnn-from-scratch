#include "ConvLayer.h"

ConvLayer::ConvLayer(int prevMatLength, int kernelLength, const MatTransformFunc& activation, const MatTransformFunc& pool) 
    : m_activation(activation), m_pool(pool) {
        initWeights(prevMatLength,kernelLength);
}

void ConvLayer::initWeights(int prevMatLength, int kernelLength) {
    m_kernels = Eigen::MatrixXf::Random(kernelLength,kernelLength);
    m_biases = Eigen::MatrixXf::Random(1+prevMatLength-kernelLength);
}

void ConvLayer::forwardProp(Eigen::MatrixXf X) {}

void ConvLayer::backProp() {}
    
void ConvLayer::gradDesc() {}

Eigen::MatrixXf ConvLayer::getA() {
    return m_A;
}

std::pair<int,int> ConvLayer::getOutputSize(std::pair<int,int> inputDims) {

    Eigen::MatrixXf input = Eigen::MatrixXf::Random(inputDims.first,inputDims.second);
    Eigen::MatrixXf Z = m_kernels * input + m_biases;
    Eigen::MatrixXf A = m_activation(m_Z);
    Eigen::MatrixXf P = m_pool(m_A);

    int rows = P.rows();
    int cols = P.cols();
    return std::pair(cols,rows);
}


