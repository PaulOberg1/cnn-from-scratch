#ifndef NETWORK_H
#define NETWORK_H

#include "ConvLayer.h"
#include "DenseLayer.h"
#include "LayerData.h"

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

    Eigen::MatrixXf m_X;
    Eigen::MatrixXf m_Y;

public:
    Network(const LayerData& layerData, Eigen::MatrixXf X, Eigen::MatrixXf Y);

    const Eigen::MatrixXf& run(int numIterations, int learningRate);

    Eigen::MatrixXf forwardProp();

    void backProp();

    void gradDesc(int learningRate);
};

#endif