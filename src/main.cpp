#include <iostream>
#include <Eigen/Dense>

#include "Network.h"
#include "LayerData.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"

int main() {
    ConvLayerData c1Data;
    c1Data.activation = ReLU;
    c1Data.activationDeriv = deriveReLU;
    c1Data.pool = maxPool;
    c1Data.poolDeriv = deriveMaxPool;
    c1Data.kernelSize=3;

    ConvLayerData c2Data;
    c2Data.activation = ReLU;
    c2Data.activationDeriv = deriveReLU;
    c2Data.pool = maxPool;
    c2Data.poolDeriv = deriveMaxPool;
    c2Data.kernelSize=3;

    DenseLayerData d1Data;
    d1Data.activation = ReLU;
    d1Data.activationDeriv = deriveReLU;
    d1Data.numNodes = 16;

    DenseLayerData d2Data;
    d2Data.activation = sigmoid;
    d2Data.numNodes = 1;

    LayerData layerData;
    layerData.c1Data = c1Data;
    layerData.c2Data = c2Data;
    layerData.d1Data = d1Data;
    layerData.d2Data = d2Data;

    Eigen::MatrixXf X = (Eigen::MatrixXf::Random(66,66).array()+1.0f)/2.0f;
    Eigen::MatrixXf Y(1,1);
    Y << 0.943;
    try{
        Network CNN(layerData,X,Y);

        Eigen::MatrixXf y_pred = CNN.run(100,0.1);
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in Network constructor or run method: "<<e.what();
    }

    return 0;
}