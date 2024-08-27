#include <iostream>
#include <Eigen/Dense>

#include "trainMain.h"
#include "testMain.h"
#include "Network.h"
#include "LayerData.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"
#include <omp.h>


int main() {    

    ConvLayerData c1Data;
    c1Data.activation = ReLU3D;
    c1Data.activationDeriv = deriveReLU3D;
    c1Data.pool = maxPool;
    c1Data.poolDeriv = deriveMaxPool;
    c1Data.kernelDimensions=std::vector<int>({32,3,3,3});

    ConvLayerData c2Data;
    c2Data.activation = ReLU3D;
    c2Data.activationDeriv = deriveReLU3D;
    c2Data.pool = maxPool;
    c2Data.poolDeriv = deriveMaxPool;
    c2Data.kernelDimensions=std::vector<int>({64,32,3,3});

    DenseLayerData d1Data;
    d1Data.activation = ReLU;
    d1Data.activationDeriv = deriveReLU;
    d1Data.numNodes = 16;

    DenseLayerData d2Data;
    d2Data.activation = sigmoid;
    d2Data.activationDeriv = deriveSigmoid;
    d2Data.numNodes = 1;

    LayerData layerData;
    layerData.c1Data = c1Data;
    layerData.c2Data = c2Data;
    layerData.d1Data = d1Data;
    layerData.d2Data = d2Data;

    std::vector<Eigen::MatrixXf> X = {};
    for (int i=0; i<3; i++) {
        Eigen::MatrixXf subMat = Eigen::MatrixXf::Random(66,66).array()+1.0f/2.0f;
        X.push_back(subMat);
    }
    Eigen::MatrixXf Y(1,1);
    Y << 0.8109;
    try{
        std::vector<int> inputMatDimensions({3,66,66});
        Network CNN = Network(layerData,inputMatDimensions);
        trainMain("C:/EggSpector/data",CNN,inputMatDimensions);
        //CNN.storeData();
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in Network constructor or run method: "<<e.what();
    }

    return 0;
}