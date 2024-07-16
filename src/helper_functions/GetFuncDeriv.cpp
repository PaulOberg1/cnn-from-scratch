#include "helper_functions/GetFuncDeriv.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"

template<typename... Args>
const MatTransformFunc<Args...>& getFuncDeriv(const MatTransformFunc<Args...>& func) {
    std::unordered_map<MatTransformFunc<Args...>, MatTransformFunc<Args...>> funcToDeriv{
        {sigmoid,deriveSigmoid},
        {ReLU,deriveReLU},
        {avgPool,deriveAvgPool},
        {maxPool,deriveMaxPool}
    };
    return funcToDeriv[func];
}

