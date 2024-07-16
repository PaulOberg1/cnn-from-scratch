#include "helper_functions/GetFuncDeriv.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"

const MatTransformFunc& getFuncDeriv(const MatTransformFunc& func) {
    return funcToDeriv[func];
}

std::unordered_map<MatTransformFunc, MatTransformFunc> funcToDeriv{
    {sigmoid,deriveSigmoid},
    {ReLU,deriveReLU},
    {avgPool,deriveAvgPool},
    {maxPool,deriveMaxPool}
};