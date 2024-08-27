#ifndef GET_FUNC_DERIV
#define GET_FUNC_DERIV

#include <Eigen/Dense>
#include <unordered_map>

using ActivationFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;
using ActivationFuncDeriv = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)>;

using ActivationFunc3D = std::function<std::vector<Eigen::MatrixXf>(const std::vector<Eigen::MatrixXf>&)>;
using ActivationFunc3DDeriv = std::function<std::vector<Eigen::MatrixXf>(const std::vector<Eigen::MatrixXf>&, const std::vector<Eigen::MatrixXf>&)>;

using PoolFunc = std::function<std::vector<Eigen::MatrixXf>(const std::vector<Eigen::MatrixXf>&, int, int)>;
using PoolFuncDeriv = std::function<std::vector<Eigen::MatrixXf>(const std::vector<Eigen::MatrixXf>&, const std::vector<Eigen::MatrixXf>&, int, int)>;

#endif