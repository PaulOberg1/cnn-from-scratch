#ifndef POOL_FUNCS_H
#define POOL_FUNCS_H

#include <Eigen/Dense>

Eigen::MatrixXf avgPool(const Eigen::MatrixXf& mat, int stride=2, int poolSize=2);

Eigen::MatrixXf maxPool(const Eigen::MatrixXf& mat, int stride=2, int poolSize=2);

Eigen::MatrixXf deriveAvgPool(const Eigen::MatrixXf& mat, int stride=2, int poolSize=2);

Eigen::MatrixXf deriveMaxPool(const Eigen::MatrixXf& mat, int stride=2, int poolSize=2);


#endif