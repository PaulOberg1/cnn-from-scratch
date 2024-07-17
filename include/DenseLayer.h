#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "helper_functions/GetFuncDeriv.h"

#include <Eigen/Dense>


using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

class DenseLayer{
private:
    Eigen::MatrixXf m_weights;
    Eigen::MatrixXf m_biases;
    ActivationFunc m_activation;

    Eigen::MatrixXf m_Z;
    Eigen::MatrixXf m_A;

    Eigen::MatrixXf m_dZ;
    Eigen::MatrixXf m_dA;
    Eigen::MatrixXf m_dW;
    Eigen::MatrixXf m_dB;

public:
    DenseLayer(int prevLayerNodes, int curLayerNodes, const ActivationFunc& activation);

    void initWeights(int prevLayerNodes, int curLayerNodes);

    void forwardProp(Eigen::MatrixXf X);

    void backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, Eigen::MatrixXf prevLayerA);
    void backProp(Eigen::MatrixXf Y, Eigen::MatrixXf prevLayerA);

    void gradDesc(int learningRate);

    Eigen::MatrixXf getZ();
    Eigen::MatrixXf getA();
    Eigen::MatrixXf getW();

    Eigen::MatrixXf getDz();
};

#endif