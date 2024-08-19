#include "DenseLayer.h"
#include <iostream>

DenseLayer::DenseLayer(int prevLayerNodes, int curLayerNodes, const ActivationFunc& activation) : m_activation(activation) {
    initWeights(prevLayerNodes,curLayerNodes);
}

void DenseLayer::initWeights(int prevLayerNodes, int curLayerNodes) {
    assert(curLayerNodes>0 && prevLayerNodes>0);
    m_weights = ((Eigen::MatrixXf::Random(curLayerNodes,prevLayerNodes).array()+1.0f)/2.0f) * std::sqrt(2.0f / curLayerNodes);
    m_biases = Eigen::MatrixXf(curLayerNodes,1);
    m_biases.setZero();
}

void DenseLayer::forwardProp(Eigen::MatrixXf X) {
    assert(m_weights.cols()==X.rows());
    try {
        m_Z = m_weights * X + m_biases;
        m_A = m_activation(m_Z);
        
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in DenseLayer::forwardProp: "<<e.what();
    }
}

void DenseLayer::backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, Eigen::MatrixXf prevLayerA) {
    try{
        m_dA = nextLayerDz(0,0) * nextLayerW.array();
        m_dZ = getActFuncDeriv(m_activation)(m_A, m_dA.transpose());
        m_dW = m_dZ * prevLayerA.transpose();
        
        m_dB = m_dZ;
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in DenseLayer::backProp: "<<e.what();
    }
}

void DenseLayer::backProp(Eigen::MatrixXf Y, Eigen::MatrixXf prevLayerA) {
    try { 

        m_dZ = m_A - Y;
        m_dW = m_dZ(0,0) * prevLayerA.transpose();
        m_dB = m_dZ;
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in DenseLayer::backProp: "<<e.what();
    }
}

void DenseLayer::gradDesc(double learningRate) {
    
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