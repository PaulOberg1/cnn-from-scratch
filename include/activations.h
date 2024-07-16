#include <Eigen/Dense>
#include <unordered_map>
#include <string>

using MatTransformFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

const Eigen::MatrixXf& sigmoid(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& ReLU(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& deriveSigmoid(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& deriveReLU(const Eigen::MatrixXf& mat);

const MatTransformFunc& getInverse(const MatTransformFunc& func);

std::unordered_map<MatTransformFunc, MatTransformFunc> functionToInverseMap{
    {sigmoid,deriveSigmoid},
    {ReLU,deriveReLU}
};