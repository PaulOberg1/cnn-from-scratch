#ifndef ACTIVATION_FUNCS_H
#define ACTIVATION_FUNCS_H

#include <Eigen/Dense>

const Eigen::MatrixXf& sigmoid(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& ReLU(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& deriveSigmoid(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& deriveReLU(const Eigen::MatrixXf& mat);

#endif