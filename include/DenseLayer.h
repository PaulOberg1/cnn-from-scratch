#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <Eigen/Dense>

class DenseLayer{
private:
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;

public:
    DenseLayer(int prevLayerNodes, int curLayerNodes);

    void initWeights(int prevLayerNodes, int curLayerNodes);

    void forwardProp();

    void backProp();

    void gradDesc();
    
};

#endif