#include "helper_functions/GetFuncDeriv.h"
#include "helper_functions/ActivationFuncs.h"
#include "helper_functions/PoolFuncs.h"

struct FuncHasher {
    template <typename T>
    std::size_t operator()(const T& func) const {
        return std::hash<const void*>()(func.target<void>());
    }
};

struct FuncEqual {
    template <typename T>
    bool operator()(const T& func1, const T& func2) const {
        return func1.target<void>() == func2.target<void>();
    }

};

ActivationFunc getActFuncDeriv(const ActivationFunc& func) {
    static std::unordered_map<ActivationFunc, ActivationFunc, FuncHasher, FuncEqual> actFuncDerivMap{
        {sigmoid,deriveSigmoid},
        {ReLU,deriveReLU}
    };
    return actFuncDerivMap[func];
}

PoolFunc getPoolFuncDeriv(const PoolFunc& func) {
    static std::unordered_map<PoolFunc, PoolFunc, FuncHasher, FuncEqual> poolFuncDerivMap{
        {avgPool,deriveAvgPool},
        {maxPool,deriveMaxPool}
    };
    return poolFuncDerivMap[func];
}
