#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "helper_functions/GetFuncDeriv.h"

#include <Eigen/Dense>

class ConvLayer{
private:
    std::vector<std::vector<Eigen::MatrixXf>> m_kernels;
    Eigen::VectorXf m_biases;
    
    ActivationFunc3D m_activation;
    ActivationFunc3DDeriv m_activationDeriv;
    PoolFunc m_pool;
    PoolFuncDeriv m_poolDeriv;

    std::vector<int> m_outputSize;

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
    ConvLayer(std::vector<int> prevMatDimensions, std::vector<int> kernelDimensions, const ActivationFunc3D& activation, const ActivationFunc3DDeriv& activationDeriv, const PoolFunc& pool, const PoolFuncDeriv& poolDeriv, int poolStride, int poolSize);

    void initWeights(std::vector<int> prevMatDimensions, std::vector<int> kernelDimensions);

    void forwardProp(std::vector<Eigen::MatrixXf> X);

    void backProp(std::vector<std::vector<Eigen::MatrixXf>> nextLayerW, std::vector<Eigen::MatrixXf> nextLayerDz, std::vector<Eigen::MatrixXf> layerInputMat);
    void backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, std::vector<Eigen::MatrixXf> layerInputMat);

    void gradDesc(double learningRate);
    
    std::vector<Eigen::MatrixXf> getZ();
    std::vector<Eigen::MatrixXf> getA();
    std::vector<Eigen::MatrixXf> getP();
    std::vector<std::vector<Eigen::MatrixXf>> getK();

    Eigen::VectorXf getFlattenedP();

    std::vector<Eigen::MatrixXf> getDz();

    std::vector<int> getOutputSize();

    void calcOutputSize(std::vector<int> prevMatDimensions,std::vector<int> kernelDimensions);

    std::vector<Eigen::MatrixXf> convolve(const std::vector<Eigen::MatrixXf>& mat, const std::vector<std::vector<Eigen::MatrixXf>>& kernels, const Eigen::VectorXf& biases);
    std::vector<Eigen::MatrixXf> convolve(const std::vector<Eigen::MatrixXf>& mat, const std::vector<std::vector<Eigen::MatrixXf>>& kernels);
    Eigen::MatrixXf convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, int padding);
    Eigen::MatrixXf convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, const Eigen::MatrixXf biases, int padding);

    std::vector<Eigen::MatrixXf> reshapeTo3D(const Eigen::MatrixXf& oldMat, const std::vector<Eigen::MatrixXf>& newMat);

    //void storeData(std::string path);
};
#endif