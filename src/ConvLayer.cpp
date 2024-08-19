#include "ConvLayer.h"
#include <iostream>

ConvLayer::ConvLayer(int prevMatLength, int kernelLength, const ActivationFunc& activation, const ActivationFuncDeriv& activationDeriv, const PoolFunc& pool, const PoolFuncDeriv& poolDeriv, int poolStride, int poolSize)
    : m_activation(activation), m_activationDeriv(activationDeriv), m_pool(pool), m_poolDeriv(poolDeriv), m_poolStride(poolStride), m_poolSize(poolSize) {
        initWeights(prevMatLength,kernelLength);
}

void ConvLayer::initWeights(int prevMatLength, int kernelLength) {
    assert(kernelLength>0 && 1+prevMatLength-kernelLength>0);
    
    m_kernels = (Eigen::MatrixXf::Random(kernelLength,kernelLength).array()+1.0f)/2.0f;
    m_biases = Eigen::MatrixXf(1+prevMatLength-kernelLength,1+prevMatLength-kernelLength);
    m_biases.setZero();
    calcOutputSize(prevMatLength);
    m_kernels = m_kernels.array() * std::sqrt(2.0f / getOutputSize());
}

void ConvLayer::forwardProp(Eigen::MatrixXf X) {
    try{
        m_Z = convolve(X, m_kernels, m_biases,0);
        m_A = m_activation(m_Z);
        m_P = m_pool(m_A, m_poolStride, m_poolSize);
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in ConvLayer::forwardProp : "<<e.what();
    }
}

void ConvLayer::backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, Eigen::MatrixXf layerInputMat, bool prevLayerConv) {
    try{
        if (prevLayerConv) {
            Eigen::MatrixXf temp_m_dP = nextLayerW.transpose() * nextLayerDz;
            Eigen::Map<Eigen::MatrixXf> reshaped(temp_m_dP.data(), m_P.rows(), m_P.cols());
            m_dP = reshaped;
        }
        else {
            m_dP = convolve(nextLayerDz,nextLayerW,m_kernels.rows()-1);
        }
        m_dA = m_poolDeriv(m_A, m_dP, m_poolStride, m_poolSize);

        m_dZ = m_activationDeriv(m_Z,m_dA);

        m_dK = convolve(layerInputMat,m_dZ,0);
        m_dB = m_dZ;
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in ConvLayer::backProp: "<<e.what();
    }
}
    
void ConvLayer::gradDesc(double learningRate) {
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

int ConvLayer::getOutputSize() {
    return m_outputSize;
}

Eigen::MatrixXf ConvLayer::getFlattenedP() {
    Eigen::VectorXf vec = Eigen::Map<Eigen::VectorXf>(m_P.data(),m_P.size());
    Eigen::MatrixXf mat = vec;
    return mat;
}

void ConvLayer::calcOutputSize(int prevMatLength) {
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(prevMatLength,prevMatLength);
    Eigen::MatrixXf Z = convolve(input,m_kernels,m_biases,0);
    Eigen::MatrixXf A = m_activation(Z);
    Eigen::MatrixXf P = m_pool(A, m_poolStride, m_poolSize);

    m_outputSize = P.rows();
}

Eigen::MatrixXf ConvLayer::convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, int padding) {

    int rows = inputMat.rows()+2*padding;
    int cols = inputMat.cols()+2*padding;
    assert(rows>0 && cols>0);

    int kRows = grad.rows();
    int kCols = grad.cols();
    assert(kRows>0 && kCols>0);

    int newRows = 1+rows-kRows;
    int newCols = 1+cols-kCols;
    assert(newRows>0 && newCols>0);

    Eigen::MatrixXf mat (rows, cols);
    mat.setZero();
    mat.block(padding,padding,rows-2*padding,cols-2*padding) = inputMat;

    Eigen::MatrixXf returnMat (newRows, newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            assert(i+kRows<=rows && j+kCols<=cols);
            returnMat(i, j) = (mat.block(i,j,kRows,kCols).array() * grad.array()).sum();
        }
    }
    return returnMat;
}

Eigen::MatrixXf ConvLayer::convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, const Eigen::MatrixXf biases, int padding) {
    int rows = inputMat.rows()+2*padding;
    int cols = inputMat.cols()+2*padding;
    assert(rows>0 && cols>0);

    int kRows = grad.rows();
    int kCols = grad.cols();
    assert(kRows>0 && kCols>0);

    int newRows = biases.rows();
    int newCols = biases.cols();
    assert(newRows>0 && newCols>0);

    assert(biases.rows()==1+rows-kRows && biases.cols()==1+cols-kCols);

    Eigen::MatrixXf mat(rows,cols);
    mat.setZero();
    mat.block(padding,padding,rows-2*padding,cols-2*padding) = inputMat;

    Eigen::MatrixXf returnMat (newRows, newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            assert(i+kRows<=rows && j+kCols<=cols);
            returnMat(i, j) = (mat.block(i,j,kRows,kCols).array() * grad.array()).sum() + biases(i, j);
        }
    }
    return returnMat;
}
