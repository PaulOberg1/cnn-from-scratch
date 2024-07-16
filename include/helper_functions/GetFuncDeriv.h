#ifndef GET_FUNC_DERIV
#define GET_FUNC_DERIV

#include <unordered_map>
#include <string>
#include <functional>
#include <Eigen/Dense>

using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

const MatTransformFunc& getFuncDeriv(const MatTransformFunc& func);

#endif