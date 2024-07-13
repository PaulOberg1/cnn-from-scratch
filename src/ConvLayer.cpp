#include <Eigen/Dense>

class ConvLayer{
    private:

    Eigen::MatrixXf kernels;
    Eigen::MatrixXf biases;

    public:
    ConvLayer(int prevMatLength, int kernelLength) {
        initWeights(prevMatLength,kernelLength);
    }

    void initWeights(int prevMatLength, int kernelLength) {
        kernels = Eigen::MatrixXf::Random(kernelLength,kernelLength);
        biases = Eigen::MatrixXf::Random(1+prevMatLength-kernelLength);
    }
};