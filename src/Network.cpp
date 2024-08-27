#include "Network.h"
#include <iostream>
#include <numeric>

Network::Network(const LayerData& layerData, std::vector<int> inputMatDimensions) {
    ConvLayerData c1Data = layerData.c1Data;
    ConvLayerData c2Data = layerData.c2Data;

    DenseLayerData d1Data = layerData.d1Data;
    DenseLayerData d2Data = layerData.d2Data;
    
    m_convLayer1 = std::make_unique<ConvLayer>(inputMatDimensions, c1Data.kernelDimensions, c1Data.activation, c1Data.activationDeriv, c1Data.pool, c1Data.poolDeriv, 2, 2);    
    m_convLayer2 = std::make_unique<ConvLayer>(m_convLayer1->getOutputSize(), c2Data.kernelDimensions, c2Data.activation, c2Data.activationDeriv, c2Data.pool, c2Data.poolDeriv, 2, 2);

    std::vector<int> outputSize = m_convLayer2->getOutputSize();
    int flattenedOutputSize = std::accumulate(outputSize.begin(),outputSize.end(),1.0f,[](int a, int b) {return a*b; });

    m_denseLayer1 = std::make_unique<DenseLayer>(flattenedOutputSize, d1Data.numNodes, d1Data.activation, d1Data.activationDeriv);
    m_denseLayer2 = std::make_unique<DenseLayer>(d1Data.numNodes, d2Data.numNodes, d2Data.activation, d2Data.activationDeriv);
}

Eigen::MatrixXf Network::run(int numIterations, double learningRate, std::vector<Eigen::MatrixXf> X, Eigen::MatrixXf Y) const {
    for (int i=0; i<numIterations; i++) {
        Eigen::MatrixXf y_pred = forwardProp(X);
        backProp(X,Y);
        gradDesc(learningRate);
    }
    std::cout<<"\n\n\n";
    return forwardProp(X);
}

Eigen::MatrixXf Network::forwardProp(std::vector<Eigen::MatrixXf> X) const {
    m_convLayer1->forwardProp(X);
    m_convLayer2->forwardProp(m_convLayer1->getP());
    m_denseLayer1->forwardProp(m_convLayer2->getFlattenedP());
    m_denseLayer2->forwardProp(m_denseLayer1->getA());

    return m_denseLayer2->getA();
}

void Network::backProp(std::vector<Eigen::MatrixXf> X, Eigen::MatrixXf Y) const {

    m_denseLayer2->backProp(Y, m_denseLayer1->getA());

    m_denseLayer1->backProp(m_denseLayer2->getW(), m_denseLayer2->getDz(), m_convLayer2->getFlattenedP());

    m_convLayer2->backProp(m_denseLayer1->getW(),m_denseLayer1->getDz(),m_convLayer1->getP());

    m_convLayer1->backProp(m_convLayer2->getK(), m_convLayer2->getDz(), X);
}

void Network::gradDesc(double learningRate) const {
    m_convLayer1->gradDesc(learningRate);
    m_convLayer2->gradDesc(learningRate);
    m_denseLayer1->gradDesc(learningRate);
    m_denseLayer2->gradDesc(learningRate);
}

/*
void Network::storeData() const {
    m_convLayer1->storeData("./network_data");
    m_convLayer2->storeData("./network_data");
    m_denseLayer1->storeData("./network_data");
    m_denseLayer2->storeData("./network_data");
}
*/