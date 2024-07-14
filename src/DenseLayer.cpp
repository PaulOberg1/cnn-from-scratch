#include "DenseLayer.h"

DenseLayer::DenseLayer(int prevLayerNodes, int curLayerNodes) {
    initWeights(prevLayerNodes,curLayerNodes);
}

void DenseLayer::initWeights(int prevLayerNodes, int curLayerNodes) {
    weights = Eigen::MatrixXf::Random(curLayerNodes,prevLayerNodes);
    biases = Eigen::MatrixXf::Random(curLayerNodes);
}

void DenseLayer::forwardProp(Eigen::MatrixXf X) {}

void DenseLayer::backProp() {}

void DenseLayer::gradDesc() {}

Eigen::MatrixXf DenseLayer::getA() {}