#include "helper_functions/ActivationFuncs.h"

Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& mat) {
    Eigen::MatrixXf returnMat;
    returnMat.resize(mat.rows(), mat.cols());
    for (int i=0; i<returnMat.rows(); i++) {
        for (int j=0; j<returnMat.cols(); j++) {
            returnMat(i,j) = 1 / (1 + std::exp(-mat(i,j)));
        }
    }
    return returnMat;
}

Eigen::MatrixXf ReLU(const Eigen::MatrixXf& mat) {
    Eigen::MatrixXf returnMat;
    returnMat.resize(mat.rows(), mat.cols());
    for (int i=0; i<returnMat.rows(); i++) {
        for (int j=0; j<returnMat.cols(); j++) {
            returnMat(i, j) = mat(i,j) > 0 ? mat(i,j) : 0;
        }
    }
    return returnMat;
}

Eigen::MatrixXf deriveSigmoid(const Eigen::MatrixXf& mat) {
    Eigen::MatrixXf sig = sigmoid(mat);
    return sig.array() * (1.0f - sig.array());
}

Eigen::MatrixXf deriveReLU(const Eigen::MatrixXf& mat) {
    int cols = mat.cols();
    int rows = mat.rows();
    Eigen::MatrixXf outMat(cols,rows);

    for (int i=0; i<cols; i++) {
        for (int j=0; j<rows; j++)
            outMat(i,j) = mat(i,j) > 0;
    }
    return outMat;
}

