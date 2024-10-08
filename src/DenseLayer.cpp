#include "DenseLayer.h"
#include <iostream>
#include <fstream>

DenseLayer::DenseLayer(int prevLayerNodes, int curLayerNodes, const ActivationFunc& activation, const ActivationFuncDeriv& activationDeriv) 
    : m_activation(activation), m_activationDeriv(activationDeriv) {
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
        m_dZ = m_activationDeriv(m_A, m_dA.transpose());
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

void DenseLayer::storeData(std::string path) {
    std::ofstream file(path,std::ios::binary);

    if (file.is_open()) {
        int w_rows = m_weights.rows();
        int w_cols = m_weights.cols();

        file.write(reinterpret_cast<char*>(&w_rows),sizeof(int));
        file.write(reinterpret_cast<char*>(&w_cols),sizeof(int));
        file.write(reinterpret_cast<const char*>(m_weights.data()),w_rows*w_cols*sizeof(float));


        int b_rows = m_biases.rows();
        int b_cols = m_biases.cols();

        file.write(reinterpret_cast<char*>(&b_rows),sizeof(int));
        file.write(reinterpret_cast<char*>(&b_cols),sizeof(int));
        file.write(reinterpret_cast<const char*>(m_biases.data()),b_rows*b_cols*sizeof(float));

        file.close();
    }
}