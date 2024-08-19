#ifndef GET_FUNC_DERIV
#define GET_FUNC_DERIV

#include <Eigen/Dense>
#include <unordered_map>

using ActivationFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;
using ActivationFuncDeriv = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)>;
using PoolFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, int, int)>;
using PoolFuncDeriv = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&, int, int)>;

ActivationFuncDeriv getActFuncDeriv(const ActivationFunc& func);
PoolFuncDeriv getPoolFuncDeriv(const PoolFunc& func);

#endif