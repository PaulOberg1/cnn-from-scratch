#ifndef LAYER_DATA_H
#define LAYER_DATA_H

#include <Eigen/Dense>
#include <vector>

using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

struct ConvLayerData{
    MatTransformFunc activation;
    MatTransformFunc pool;
    int kernelSize;
};

struct DenseLayerData{
    MatTransformFunc activation;
    int numNodes;
};

struct LayerData{
    ConvLayerData c1Data;
    ConvLayerData c2Data;

    DenseLayerData d1Data;
    DenseLayerData d2Data;
};

#endif