

#include <iostream>
#include <Eigen/Dense>

#include "trainMain.h"
#include "testMain.h"
#include "Network.h"
#include "LayerData.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"
#include <omp.h>



int runNetwork(std::string path) {
    ConvLayerData c1Data;
    c1Data.activation = LeakyReLU3D;
    c1Data.activationDeriv = deriveLeakyReLU3D;
    c1Data.pool = maxPool;
    c1Data.poolDeriv = deriveMaxPool;
    c1Data.kernelDimensions=std::vector<int>({32,3,3,3});

    ConvLayerData c2Data;
    c2Data.activation = LeakyReLU3D;
    c2Data.activationDeriv = deriveLeakyReLU3D;
    c2Data.pool = maxPool;
    c2Data.poolDeriv = deriveMaxPool;
    c2Data.kernelDimensions=std::vector<int>({64,32,3,3});

    DenseLayerData d1Data;
    d1Data.activation = LeakyReLU;
    d1Data.activationDeriv = deriveLeakyReLU;
    d1Data.numNodes = 64;

    DenseLayerData d2Data;
    d2Data.activation = LeakyReLU;
    d2Data.activationDeriv = deriveLeakyReLU;
    d2Data.numNodes = 16;

    DenseLayerData d3Data;
    d3Data.activation = sigmoid;
    d3Data.activationDeriv = deriveSigmoid;
    d3Data.numNodes = 1;

    LayerData layerData;
    layerData.c1Data = c1Data;
    layerData.c2Data = c2Data;
    layerData.d1Data = d1Data;
    layerData.d2Data = d2Data;
    layerData.d3Data = d3Data;

    std::vector<int> inputMatDimensions({3,66,66});
    Network CNN = Network(layerData,inputMatDimensions);

    int res = testImg(path,CNN,inputMatDimensions);
    return res==1 ? true : false;
}


extern "C" {
    __declspec(dllexport) int simpleFunction(const char* path) {
        return runNetwork(path); // Just return a constant integer
    }
}


int main() {    

    ConvLayerData c1Data;
    c1Data.activation = LeakyReLU3D;
    c1Data.activationDeriv = deriveLeakyReLU3D;
    c1Data.pool = maxPool;
    c1Data.poolDeriv = deriveMaxPool;
    c1Data.kernelDimensions=std::vector<int>({32,3,3,3});

    ConvLayerData c2Data;
    c2Data.activation = LeakyReLU3D;
    c2Data.activationDeriv = deriveLeakyReLU3D;
    c2Data.pool = maxPool;
    c2Data.poolDeriv = deriveMaxPool;
    c2Data.kernelDimensions=std::vector<int>({64,32,3,3});

    DenseLayerData d1Data;
    d1Data.activation = LeakyReLU;
    d1Data.activationDeriv = deriveLeakyReLU;
    d1Data.numNodes = 64;

    DenseLayerData d2Data;
    d2Data.activation = LeakyReLU;
    d2Data.activationDeriv = deriveLeakyReLU;
    d2Data.numNodes = 16;

    DenseLayerData d3Data;
    d3Data.activation = sigmoid;
    d3Data.activationDeriv = deriveSigmoid;
    d3Data.numNodes = 1;

    LayerData layerData;
    layerData.c1Data = c1Data;
    layerData.c2Data = c2Data;
    layerData.d1Data = d1Data;
    layerData.d2Data = d2Data;
    layerData.d3Data = d3Data;

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