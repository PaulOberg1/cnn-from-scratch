#include <map>
#include <string>
#include <utility>
#include <Eigen/Dense>

class Network{
    private:
    Eigen::MatrixXf convKernels;
    Eigen::MatrixXf convBiases;

    Eigen::MatrixXf denseWeights;
    Eigen::MatrixXf denseBiases;

    public:
    Network(std::map<std::string,std::map<std::string,std::pair<int,int>>> dimensions) {
        std::pair<float,float> ConvKernelDims = dimensions.at("ConvLayer").at("Kernels");
        std::pair<float,float> ConvBiasDims = dimensions.at("ConvLayer").at("Biases");

        std::pair<float,float> DenseWeightDims = dimensions.at("DenseLayer").at("Kernels");
        std::pair<float,float> DenseBiasDims = dimensions.at("DenseLayer").at("Biases");

        convKernels = Eigen::MatrixXf::Random(ConvKernelDims.first,ConvKernelDims.second);
        convBiases = Eigen::MatrixXf::Random(ConvBiasDims.first,ConvBiasDims.second);

        denseWeights = Eigen::MatrixXf::Random(DenseWeightDims.first,DenseWeightDims.second);
        denseBiases = Eigen::MatrixXf::Random(DenseBiasDims.first,DenseBiasDims.second);

    }
    void run() {}
    void forwardProp() {}
    void backProp() {}
    void gradDesc() {}
};