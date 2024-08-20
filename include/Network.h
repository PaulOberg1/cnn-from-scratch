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
    std::unique_ptr<ConvLayer> m_convLayer1;
    std::unique_ptr<ConvLayer> m_convLayer2;
    std::unique_ptr<DenseLayer> m_denseLayer1;
    std::unique_ptr<DenseLayer> m_denseLayer2;

    int m_Xrows;
    int m_Yrows;

public:
    Network(const LayerData& layerData, int Xrows, int Yrows);

    Eigen::MatrixXf run(int numIterations, double learningRate, Eigen::MatrixXf X, Eigen::MatrixXf Y) const;

    Eigen::MatrixXf forwardProp(Eigen::MatrixXf X) const;

    void backProp(Eigen::MatrixXf X, Eigen::MatrixXf Y) const;

    void gradDesc(double learningRate) const;
};

#endif