#include "Network.h"

Network::Network(const LayerData& layerData, Eigen::MatrixXf X, Eigen::MatrixXf Y) 
    : m_X(X), m_Y(Y) {
    ConvLayerData c1Data = layerData.c1Data;
    ConvLayerData c2Data = layerData.c2Data;

    DenseLayerData d1Data = layerData.d1Data;
    DenseLayerData d2Data = layerData.d2Data;
    
    m_convLayer1 = std::make_unique<ConvLayer>(X.rows(), c1Data.kernelSize, c1Data.activation, c1Data.pool, 2, 2);    
    m_convLayer2 = std::make_unique<ConvLayer>(m_convLayer1->getOutputSize(), c2Data.kernelSize, c2Data.activation, c2Data.pool,2, 2);

    int flattenedOutputSize = pow(m_convLayer2->getOutputSize(),2);

    m_denseLayer1 = std::make_unique<DenseLayer>(flattenedOutputSize, d1Data.numNodes, d1Data.activation);
    m_denseLayer2 = std::make_unique<DenseLayer>(d1Data.numNodes, d2Data.numNodes, d2Data.activation);
}

Eigen::MatrixXf Network::run(int numIterations, double learningRate) {
    for (int i=0; i<numIterations; i++) {
        Eigen::MatrixXf y_pred = forwardProp();
        backProp();
        gradDesc(learningRate);
    }
    return forwardProp();
}

Eigen::MatrixXf Network::forwardProp() {
    m_convLayer1->forwardProp(m_X);
    m_convLayer2->forwardProp(m_convLayer1->getP());
    m_denseLayer1->forwardProp(m_convLayer2->getFlattenedP());
    m_denseLayer2->forwardProp(m_denseLayer1->getA());

    return m_denseLayer2->getA();
}

void Network::backProp() {

    m_denseLayer2->backProp(m_Y, m_denseLayer1->getA());

    m_denseLayer1->backProp(m_denseLayer2->getW(), m_denseLayer2->getDz(), m_convLayer2->getFlattenedP());

    m_convLayer2->backProp(m_denseLayer1->getW(),m_denseLayer1->getDz(),m_convLayer1->getP(),true);

    m_convLayer1->backProp(m_convLayer2->getK(), m_convLayer2->getDz(), m_X, false);

}

void Network::gradDesc(double learningRate) {
    m_convLayer1->gradDesc(learningRate);
    m_convLayer2->gradDesc(learningRate);
    m_denseLayer1->gradDesc(learningRate);
    m_denseLayer2->gradDesc(learningRate);
}