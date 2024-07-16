#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <Eigen/Dense>

using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

class DenseLayer{
private:
    Eigen::MatrixXf m_weights;
    Eigen::MatrixXf m_biases;
    MatTransformFunc m_activation;

    Eigen::MatrixXf m_Z;
    Eigen::MatrixXf m_A;

    Eigen::MatrixXf m_dZ;
    Eigen::MatrixXf m_dA;

public:
    DenseLayer(int prevLayerNodes, int curLayerNodes, const MatTransformFunc& activation);

    void initWeights(int prevLayerNodes, int curLayerNodes);

    void forwardProp(Eigen::MatrixXf X);

    void backProp();

    void gradDesc(int learningRate);

    Eigen::MatrixXf getZ();
    Eigen::MatrixXf getA();
};

#endif