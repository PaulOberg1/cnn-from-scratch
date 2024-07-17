#ifndef ACTIVATION_FUNCS_H
#define ACTIVATION_FUNCS_H

#include <Eigen/Dense>

Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& mat);

Eigen::MatrixXf ReLU(const Eigen::MatrixXf& mat);

Eigen::MatrixXf deriveSigmoid(const Eigen::MatrixXf& mat);

Eigen::MatrixXf deriveReLU(const Eigen::MatrixXf& mat);

#endif