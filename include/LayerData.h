#ifndef LAYER_DATA_H
#define LAYER_DATA_H

#include "helper_functions/GetFuncDeriv.h"

#include <Eigen/Dense>
#include <vector>

struct ConvLayerData{
    ActivationFunc3D activation;
    ActivationFunc3DDeriv activationDeriv;
    PoolFunc pool;
    PoolFuncDeriv poolDeriv;
    std::vector<int> kernelDimensions;
};

struct DenseLayerData{
    ActivationFunc activation;
    ActivationFuncDeriv activationDeriv;
    int numNodes;
};

struct LayerData{
    ConvLayerData c1Data;
    ConvLayerData c2Data;

    DenseLayerData d1Data;
    DenseLayerData d2Data;
};

#endif