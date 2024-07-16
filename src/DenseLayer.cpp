#include "DenseLayer.h"

DenseLayer::DenseLayer(int prevLayerNodes, int curLayerNodes, const MatTransformFunc& activation) : m_activation(activation) {
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

void DenseLayer::backProp() {}

void DenseLayer::gradDesc(int learningRate) {
    m_Z -= learningRate*m_dZ;
    m_A -= learningRate*m_dA;
}

Eigen::MatrixXf DenseLayer::getZ() {
    return m_Z;
}

Eigen::MatrixXf DenseLayer::getA() {
    return m_A;
}