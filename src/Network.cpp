#include "Network.h"

Network::Network(int Conv1Nodes, int Conv2Nodes, int Dense1Nodes, int Dense2Nodes) {
    m_convLayer1 = new ConvLayer(1,Conv1Nodes);
    m_convLayer2 = new ConvLayer(Conv1Nodes,Conv2Nodes);

    m_denseLayer1 = new DenseLayer(Conv2Nodes,Dense1Nodes);
    m_denseLayer2 = new DenseLayer(Dense1Nodes, Dense2Nodes);
}

void Network::run(Eigen::MatrixXf X) {}

Eigen::MatrixXf Network::forwardProp(Eigen::MatrixXf X) {
    m_convLayer1->forwardProp(X);
    m_convLayer2->forwardProp(m_convLayer1->getA());
    m_denseLayer1->forwardProp(m_convLayer2->getA());
    m_denseLayer2->forwardProp(m_denseLayer1->getA());

    return m_denseLayer2->getA();
}

void Network::backProp() {}

void Network::gradDesc() {}