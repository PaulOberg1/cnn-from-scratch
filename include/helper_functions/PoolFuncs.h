#ifndef POOL_FUNCS_H
#define POOL_FUNCS_H

#include <Eigen/Dense>

const Eigen::MatrixXf& avgPool(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& maxPool(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& deriveAvgPool(const Eigen::MatrixXf& mat);

const Eigen::MatrixXf& deriveMaxPool(const Eigen::MatrixXf& mat);



#endif