#include <unordered_map>
#include <string>
#include <functional>
#include <Eigen/Dense>

using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

const MatTransformFunc& getInverse(const MatTransformFunc& func);