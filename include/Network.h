#ifndef NETWORK_H
#define NETWORK_H

#include "ConvLayer.h"
#include "DenseLayer.h"

#include <map>
#include <string>
#include <utility>
#include <Eigen/Dense>

class Network{
private:
    ConvLayer* m_convLayer1;
    ConvLayer* m_convLayer2;
    DenseLayer* m_denseLayer1;
    DenseLayer* m_denseLayer2;

public:
    Network(int Conv1Nodes, int Conv2Nodes, int Dense1Nodes, int Dense2Nodes);

    void run(Eigen::MatrixXf X);

    Eigen::MatrixXf forwardProp(Eigen::MatrixXf X);

    void backProp();

    void gradDesc();
};

#endif