#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "helper_functions/GetFuncDeriv.h"

#include <Eigen/Dense>

class ConvLayer{
private:
    std::vector<std::vector<Eigen::MatrixXf>> m_kernels;
    Eigen::MatrixXf m_biases;
    
    ActivationFunc3D m_activation;
    ActivationFunc3DDeriv m_activationDeriv;
    PoolFunc m_pool;
    PoolFuncDeriv m_poolDeriv;

    int m_outputSize;

    int m_poolStride;
    int m_poolSize;

    std::vector<Eigen::MatrixXf> m_Z;
    std::vector<Eigen::MatrixXf> m_A;
    std::vector<Eigen::MatrixXf> m_P;

    std::vector<Eigen::MatrixXf> m_dZ;
    std::vector<Eigen::MatrixXf> m_dA;
    std::vector<Eigen::MatrixXf> m_dP;;
    std::vector<std::vector<Eigen::MatrixXf>> m_dK;
    Eigen::VectorXf m_dB;

public:
    ConvLayer(int prevMatLength, int kernelLength, const ActivationFunc3D& activation, const ActivationFunc3DDeriv& activationDeriv, const PoolFunc& pool, const PoolFuncDeriv& poolDeriv, int poolStride=2, int poolSize=2);

    void initWeights(int prevMatLength, int kernelLength);

    void forwardProp(std::vector<Eigen::MatrixXf> X);

    void backProp(std::vector<Eigen::MatrixXf> nextLayerW, std::vector<Eigen::MatrixXf> nextLayerDz, std::vector<Eigen::MatrixXf> layerInputMat);
    void backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, std::vector<Eigen::MatrixXf> layerInputMat);

    void gradDesc(double learningRate);
    
    std::vector<Eigen::MatrixXf> getZ();
    std::vector<Eigen::MatrixXf> getA();
    std::vector<Eigen::MatrixXf> getP();
    std::vector<Eigen::MatrixXf> getK();

    Eigen::MatrixXf getFlattenedP();

    std::vector<Eigen::MatrixXf> getDz();

    int getOutputSize();

    void calcOutputSize(int prevMatLength);

    std::vector<Eigen::MatrixXf> convolve(const std::vector<Eigen::MatrixXf>& mat, const std::vector<std::vector<Eigen::MatrixXf>>& kernels, const Eigen::VectorXf& biases);
    std::vector<Eigen::MatrixXf> convolve(const std::vector<Eigen::MatrixXf>& mat, const std::vector<std::vector<Eigen::MatrixXf>>& kernels);
    Eigen::MatrixXf convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, int padding);
    Eigen::MatrixXf convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, const Eigen::MatrixXf biases, int padding);

    std::vector<Eigen::MatrixXf> reshapeTo3D(const Eigen::MatrixXf& oldMat, const std::vector<Eigen::MatrixXf>& newMat);

    void storeData(std::string path);
};
#endif