#include "DenseLayer.h"

DenseLayer::DenseLayer(int prevLayerNodes, int curLayerNodes, const ActivationFunc& activation) : m_activation(activation) {
    initWeights(prevLayerNodes,curLayerNodes);
}

void DenseLayer::initWeights(int prevLayerNodes, int curLayerNodes) {
    m_weights = Eigen::MatrixXf::Random(curLayerNodes,prevLayerNodes);
    m_biases = Eigen::MatrixXf::Random(curLayerNodes);
}

void DenseLayer::forwardProp(Eigen::MatrixXf X) {
    m_Z = m_weights * X + m_biases;
    m_A = m_activation(m_Z);
}

void DenseLayer::backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, Eigen::MatrixXf prevLayerA) {

    ActivationFunc activationDerivative = getActFuncDeriv(m_activation);

    m_dA = nextLayerW * nextLayerDz;
    m_dZ = activationDerivative(m_A) * m_dA;
    m_dW = m_dZ * prevLayerA;
    m_dB = m_dZ;
}

void DenseLayer::backProp(Eigen::MatrixXf Y, Eigen::MatrixXf prevLayerA) {
    m_dZ = m_A - Y;
    m_dW = m_dZ * prevLayerA;
    m_dB = m_dZ;
}

void DenseLayer::gradDesc(int learningRate) {
    m_weights -= learningRate*m_dW;
    m_biases -= learningRate*m_dB;
}

Eigen::MatrixXf DenseLayer::getZ() {
    return m_Z;
}

Eigen::MatrixXf DenseLayer::getA() {
    return m_A;
}

Eigen::MatrixXf DenseLayer::getW() {
    return m_weights;
}

Eigen::MatrixXf DenseLayer::getDz() {
    return m_dZ;
}