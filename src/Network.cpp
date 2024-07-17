#include "Network.h"

Network::Network(const LayerData& layerData, Eigen::MatrixXf X) {
    ConvLayerData c1Data = layerData.c1Data;
    ConvLayerData c2Data = layerData.c2Data;

    DenseLayerData d1Data = layerData.d1Data;
    DenseLayerData d2Data = layerData.d2Data;
    
    m_convLayer1 = new ConvLayer(1, c1Data.kernelSize, c1Data.activation, c1Data.pool, 2, 2);
    m_convLayer2 = new ConvLayer(1, c2Data.kernelSize, c2Data.activation, c2Data.pool,2, 2);

    std::pair<int,int> XDims(X.cols(),X.rows());
    std::pair<int,int> C1OutDims = m_convLayer1->getOutputSize(XDims);
    std::pair<int,int> C2OutDims = m_convLayer2->getOutputSize(C1OutDims);
    int flattenedOutputSize = C2OutDims.first * C2OutDims.second;

    m_denseLayer1 = new DenseLayer(flattenedOutputSize, d1Data.numNodes, d1Data.activation);
    m_denseLayer2 = new DenseLayer(d1Data.numNodes, d2Data.numNodes, d2Data.activation);
}

void Network::run(Eigen::MatrixXf X, Eigen::MatrixXf Y, int numIterations, int learningRate) {
    for (int i=0; i<numIterations; i++) {
        Eigen::MatrixXf y_pred = forwardProp(X);
        backProp(X,Y);
        gradDesc(learningRate);
    }
}

Eigen::MatrixXf Network::forwardProp(Eigen::MatrixXf X) {
    m_convLayer1->forwardProp(X);
    m_convLayer2->forwardProp(m_convLayer1->getA());
    m_denseLayer1->forwardProp(m_convLayer2->getA());
    m_denseLayer2->forwardProp(m_denseLayer1->getA());

    return m_denseLayer2->getA();
}

void Network::backProp(Eigen::MatrixXf X, Eigen::MatrixXf Y) {

    m_denseLayer2->backProp(Y,m_denseLayer2->getA());

    m_denseLayer1->backProp(m_denseLayer2->getW(), m_denseLayer2->getDz(), m_convLayer2->getA());

    m_convLayer2->backProp(m_denseLayer1->getW(),m_denseLayer1->getDz(),m_convLayer1->getA());

    m_convLayer1->backProp(m_convLayer2->getK(), m_convLayer2->getDz(), X);

}

void Network::gradDesc(int learningRate) {
    m_convLayer1->gradDesc(learningRate);
    m_convLayer2->gradDesc(learningRate);
    m_denseLayer1->gradDesc(learningRate);
    m_denseLayer2->gradDesc(learningRate);
}