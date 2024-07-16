#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <Eigen/Dense>

using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

class ConvLayer{
private:
    Eigen::MatrixXf m_kernels;
    Eigen::MatrixXf m_biases;
    MatTransformFunc m_activation;
    MatTransformFunc m_pool;

    Eigen::MatrixXf m_Z;
    Eigen::MatrixXf m_A;
    Eigen::MatrixXf m_P;

    Eigen::MatrixXf m_dZ;
    Eigen::MatrixXf m_dA;
    Eigen::MatrixXf m_dP;;

public:
    ConvLayer(int prevMatLength, int kernelLength, const MatTransformFunc& activation, const MatTransformFunc& pool);

    void initWeights(int prevMatLength, int kernelLength);

    void forwardProp(Eigen::MatrixXf X);

    void backProp();
    
    void gradDesc(int learningRate);
    
    Eigen::MatrixXf getA();

    std::pair<int,int> ConvLayer::getOutputSize(std::pair<int,int> inputDims);
};

#endif