#include "ConvLayer.h"

ConvLayer::ConvLayer(int prevMatLength, int kernelLength, const ActivationFunc& activation, const PoolFunc& pool, int poolStride=2, int poolSize=2) 
    : m_activation(activation), m_pool(pool), m_poolStride(poolStride), m_poolSize(poolSize) {
        initWeights(prevMatLength,kernelLength);
}

void ConvLayer::initWeights(int prevMatLength, int kernelLength) {
    m_kernels = Eigen::MatrixXf::Random(kernelLength,kernelLength);
    m_biases = Eigen::MatrixXf::Random(1+prevMatLength-kernelLength);
}

void ConvLayer::forwardProp(Eigen::MatrixXf X) {
    m_Z = m_kernels * X + m_biases;
    m_A = m_activation(m_Z);
    m_P = m_pool(m_A, m_poolStride, m_poolSize);
}

void ConvLayer::backProp() {}
    
void ConvLayer::gradDesc(int learningRate) {
    m_Z -= learningRate*m_dZ;
    m_A -= learningRate*m_dA;
    m_P -= learningRate*m_dP;
}

Eigen::MatrixXf ConvLayer::getZ() {
    return m_Z;
}

Eigen::MatrixXf ConvLayer::getA() {
    return m_A;
}

Eigen::MatrixXf ConvLayer::getP() {
    return m_P;
}

std::pair<int,int> ConvLayer::getOutputSize(std::pair<int,int> inputDims) {

    Eigen::MatrixXf input = Eigen::MatrixXf::Random(inputDims.first,inputDims.second);
    Eigen::MatrixXf Z = m_kernels * input + m_biases;
    Eigen::MatrixXf A = m_activation(m_Z);
    Eigen::MatrixXf P = m_pool(m_A, m_poolStride, m_poolSize);

    int rows = P.rows();
    int cols = P.cols();
    return std::pair(cols,rows);
}


