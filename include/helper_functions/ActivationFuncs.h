#ifndef ACTIVATION_FUNCS_H
#define ACTIVATION_FUNCS_H

#include <Eigen/Dense>

Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& mat);
std::vector<Eigen::MatrixXf> sigmoid(const std::vector<Eigen::MatrixXf>& mat);

Eigen::MatrixXf ReLU(const Eigen::MatrixXf& mat);
std::vector<Eigen::MatrixXf> ReLU(const std::vector<Eigen::MatrixXf>& mat);

Eigen::MatrixXf deriveSigmoid(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad);

Eigen::MatrixXf deriveReLU(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad);
std::vector<Eigen::MatrixXf> deriveReLU(const std::vector<Eigen::MatrixXf>& mat, const std::vector<Eigen::MatrixXf>& grad);

#endif