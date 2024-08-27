#ifndef POOL_FUNCS_H
#define POOL_FUNCS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

std::vector<Eigen::MatrixXf> avgPool(const std::vector<Eigen::MatrixXf>& mat, int stride=2, int poolSize=2);


std::vector<Eigen::MatrixXf> maxPool(const std::vector<Eigen::MatrixXf>& mat, int stride=2, int poolSize=2);

std::vector<Eigen::MatrixXf> deriveAvgPool(const std::vector<Eigen::MatrixXf>& mat, const std::vector<Eigen::MatrixXf>& grad, int stride=2, int poolSize=2);

std::vector<Eigen::MatrixXf> deriveMaxPool(const std::vector<Eigen::MatrixXf>& mat, const std::vector<Eigen::MatrixXf>& grad, int stride=2, int poolSize=2);

#endif