#include "ConvLayer.h"

ConvLayer::ConvLayer(int prevMatLength, int kernelLength, const ActivationFunc& activation, const PoolFunc& pool, int poolStride, int poolSize) 
    : m_activation(activation), m_pool(pool), m_poolStride(poolStride), m_poolSize(poolSize) {
        initWeights(prevMatLength,kernelLength);
}

void ConvLayer::initWeights(int prevMatLength, int kernelLength) {
    m_kernels = Eigen::MatrixXf::Random(kernelLength,kernelLength);
    m_biases = Eigen::MatrixXf::Random(1+prevMatLength-kernelLength);
}

void ConvLayer::forwardProp(Eigen::MatrixXf X) {
    m_Z = convolve(X, m_kernels, m_biases);
    m_A = m_activation(m_Z);
    m_P = m_pool(m_A, m_poolStride, m_poolSize);
}


void ConvLayer::backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, Eigen::MatrixXf layerInputMat) {

   m_dP = nextLayerW * nextLayerDz;
   m_dA = getPoolFuncDeriv(m_pool)(m_A, m_poolStride, m_poolSize).array() * m_dP.array();
   m_dZ = getActFuncDeriv(m_activation)(m_Z).array() * m_dA.array();
   m_dK = convolve(layerInputMat,m_dZ);
   m_dB = m_dK;
}
    
void ConvLayer::gradDesc(int learningRate) {
    m_kernels -= learningRate*m_dK;
    m_biases -= learningRate*m_dB;
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

Eigen::MatrixXf ConvLayer::getK() {
    return m_kernels;
}

Eigen::MatrixXf ConvLayer::getDz() {
    return m_dZ;
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

Eigen::MatrixXf ConvLayer::convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad) {
    int rows = inputMat.rows();
    int cols = inputMat.cols();

    int kRows = grad.rows();
    int kCols = grad.cols();

    int newRows = 1+rows-kRows;
    int newCols = 1+cols-kCols;

    Eigen::MatrixXf returnMat (newRows, newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            returnMat(i, j) = (inputMat.block(i,j,kRows,kCols).array() * grad.block(i,j,kRows,kCols).array()).sum();
        }
    }
    return returnMat;
}

Eigen::MatrixXf ConvLayer::convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, const Eigen::MatrixXf biases) {
    int rows = inputMat.rows();
    int cols = inputMat.cols();

    int kRows = grad.rows();
    int kCols = grad.cols();

    int newRows = biases.rows();
    int newCols = biases.cols();

    Eigen::MatrixXf returnMat (newRows, newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            returnMat(i, j) = (inputMat.block(i,j,kRows,kCols).array() * grad.block(i,j,kRows,kCols).array()).sum() + biases(i, j);
        }
    }
    return returnMat;
}
