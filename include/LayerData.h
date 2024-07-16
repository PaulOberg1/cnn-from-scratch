#ifndef LAYER_DATA_H
#define LAYER_DATA_H

#include "helper_functions/GetFuncDeriv.h"

#include <Eigen/Dense>
#include <vector>

struct ConvLayerData{
    ActivationFunc activation;
    PoolFunc pool;
    int kernelSize;
};

struct DenseLayerData{
    ActivationFunc activation;
    int numNodes;
};

struct LayerData{
    ConvLayerData c1Data;
    ConvLayerData c2Data;

    DenseLayerData d1Data;
    DenseLayerData d2Data;
};

#endif