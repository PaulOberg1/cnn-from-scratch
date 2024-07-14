#ifndef NETWORK_H
#define NETWORK_H

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
    Network(std::map<std::string,std::map<std::string,std::pair<int,int>>> dimensions) {}

    void run() {}

    void forwardProp() {}

    void backProp() {}
    
    void gradDesc() {}
};

#endif