#ifndef GET_FUNC_DERIV
#define GET_FUNC_DERIV

#include <Eigen/Dense>

template<typename... Args>
using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, Args...)>;

template<typename... Args>
MatTransformFunc<Args...> getFuncDeriv(const MatTransformFunc<Args...>& func);

#endif