#ifndef GET_FUNC_DERIV
#define GET_FUNC_DERIV

#include <Eigen/Dense>
#include <unordered_map>

using ActivationFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;
using PoolFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, int, int)>;

const ActivationFunc& getActFuncDeriv(const ActivationFunc& func);
const PoolFunc& getPoolFuncDeriv(const PoolFunc& func);

#endif