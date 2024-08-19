#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "helper_functions/GetFuncDeriv.h"

#include <Eigen/Dense>

class ConvLayer{
private:
    Eigen::MatrixXf m_kernels;
    Eigen::MatrixXf m_biases;
    int m_outputSize;
    ActivationFunc m_activation;
    PoolFunc m_pool;

    int m_poolStride;
    int m_poolSize;

    Eigen::MatrixXf m_Z;
    Eigen::MatrixXf m_A;
    Eigen::MatrixXf m_P;

    Eigen::MatrixXf m_dZ;
    Eigen::MatrixXf m_dA;
    Eigen::MatrixXf m_dP;;
    Eigen::MatrixXf m_dK;
    Eigen::MatrixXf m_dB;

public:
    ConvLayer(int prevMatLength, int kernelLength, const ActivationFunc& activation, const PoolFunc& pool, int poolStride=2, int poolSize=2);

    void initWeights(int prevMatLength, int kernelLength);

    void forwardProp(Eigen::MatrixXf X);

    void backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, Eigen::MatrixXf layerInputMat, bool prevLayerConv);
    
    void gradDesc(int learningRate);
    
    Eigen::MatrixXf getZ();
    Eigen::MatrixXf getA();
    Eigen::MatrixXf getP();
    Eigen::MatrixXf getK();

    Eigen::MatrixXf getFlattenedP();

    Eigen::MatrixXf getDz();

    int getOutputSize();

    void calcOutputSize(int prevMatLength);

    Eigen::MatrixXf convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, int padding);
    Eigen::MatrixXf convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, const Eigen::MatrixXf biases, int padding);

};
#endif