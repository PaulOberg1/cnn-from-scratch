#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <Eigen/Dense>

class ConvLayer{
private:
    Eigen::MatrixXf kernels;
    Eigen::MatrixXf biases;

public:
    ConvLayer(int prevMatLength, int kernelLength);

    void initWeights(int prevMatLength, int kernelLength);

    void forwardProp();

    void backProp();
    
    void gradDesc();
};

#endif