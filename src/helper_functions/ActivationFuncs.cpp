#include "helper_functions/ActivationFuncs.h"

Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    Eigen::MatrixXf returnMat (rows,cols);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            returnMat(i,j) = 1 / (1 + std::exp(-mat(i,j)));
        }
    }
    return returnMat;
}

Eigen::MatrixXf ReLU(const Eigen::MatrixXf& mat) {
    
    int rows = mat.rows();
    int cols = mat.cols();
    Eigen::MatrixXf returnMat (rows,cols);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            returnMat(i, j) = mat(i,j) > 0 ? mat(i,j) : 0;
        }
    }
    
    return returnMat;
}

Eigen::MatrixXf deriveSigmoid(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad) {
    Eigen::MatrixXf sig = sigmoid(mat);
    return sig.array() * (1.0f - sig.array()) * grad.array();
}

Eigen::MatrixXf deriveReLU(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad) {
    int rows = mat.rows();
    int cols = mat.cols();
    
    Eigen::MatrixXf outMat(rows,cols);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++)
            outMat(i,j) = mat(i,j) > 0;
    }
    return outMat.array() * grad.array();
}

