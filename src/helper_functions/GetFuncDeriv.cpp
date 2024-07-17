#include "helper_functions/GetFuncDeriv.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"

ActivationFunc getActFuncDeriv(const ActivationFunc& func) {
    std::unordered_map<ActivationFunc, ActivationFunc> actFuncDerivMap{
        {sigmoid,deriveSigmoid},
        {ReLU,deriveReLU}
    };
    return actFuncDerivMap[func];
}

PoolFunc getPoolFuncDeriv(const PoolFunc& func) {
    std::unordered_map<PoolFunc, PoolFunc> poolFuncDerivMap{
        {avgPool,deriveAvgPool},
        {maxPool,deriveMaxPool}
    };
    return poolFuncDerivMap[func];
}
