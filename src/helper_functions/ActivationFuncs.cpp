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
std::vector<Eigen::MatrixXf> ReLU3D(const std::vector<Eigen::MatrixXf>& mat) {
    int depth = mat.size();
    int height = mat.at(0).rows();
    int width = mat.at(0).cols();

    std::vector<Eigen::MatrixXf> returnMat(depth);

    for (int i=0; i<depth; i++)
        returnMat.at(i) = Eigen::MatrixXf(height,width);

    for (int i=0; i<depth; i++) {
        const Eigen::MatrixXf& subMat = mat.at(i);
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++)
                returnMat.at(i)(j, k) = subMat(j,k) > 0 ? subMat(j,k) : 0;
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
std::vector<Eigen::MatrixXf> deriveReLU3D(const std::vector<Eigen::MatrixXf>& mat, const std::vector<Eigen::MatrixXf>& grad) {
    int depth = 0;
    int height = mat.at(0).rows();
    int width = mat.at(0).cols();
    
    std::vector<Eigen::MatrixXf> returnMat(depth);
    for (int i=0; i<depth; i++) {
        returnMat.at(i) = Eigen::MatrixXf(height,width);
    }

    for (int i=0; i<depth; i++) {
        const Eigen::MatrixXf& subMat = mat.at(i);
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                returnMat.at(i)(j,k) = subMat(j,k) > 0 ? grad.at(i)(j,k) : 0;
            }
        }
    }
    return returnMat;
}
