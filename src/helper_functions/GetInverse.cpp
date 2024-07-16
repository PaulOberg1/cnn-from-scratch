#include "helper_functions/GetInverse.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"

const MatTransformFunc& getInverse(const MatTransformFunc& func) {
    return functionToInverseMap[func];
}

std::unordered_map<MatTransformFunc, MatTransformFunc> functionToInverseMap{
    {sigmoid,deriveSigmoid},
    {ReLU,deriveReLU},
    {avgPool,deriveAvgPool},
    {maxPool,deriveMaxPool}
};