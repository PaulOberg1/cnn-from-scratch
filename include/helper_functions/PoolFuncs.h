#ifndef POOL_FUNCS_H
#define POOL_FUNCS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::MatrixXf avgPool(const Eigen::MatrixXf& mat, int stride=2, int poolSize=2);
std::vector<Eigen::MatrixXf> avgPool(const std::vector<Eigen::MatrixXf>& mat, int stride=2, int poolSize=2);


Eigen::MatrixXf maxPool(const Eigen::MatrixXf& mat, int stride=2, int poolSize=2);
std::vector<Eigen::MatrixXf> maxPool(const std::vector<Eigen::MatrixXf>& mat, int stride=2, int poolSize=2);

Eigen::MatrixXf deriveAvgPool(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad, int stride=2, int poolSize=2);

Eigen::MatrixXf deriveMaxPool(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad, int stride=2, int poolSize=2);

#endif