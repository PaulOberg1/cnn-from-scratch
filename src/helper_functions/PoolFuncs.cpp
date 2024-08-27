#include "helper_functions/PoolFuncs.h"

std::vector<Eigen::MatrixXf> avgPool(const std::vector<Eigen::MatrixXf>& mat, int stride, int poolSize) {
    int depth = mat.size();
    int height = mat.at(0).rows();
    int width = mat.at(0).cols();

    int newHeight = height/stride;
    int newWidth = width/stride;

    std::vector<Eigen::MatrixXf> returnMat(depth);
    for (int i = 0; i < depth; ++i) {
        returnMat.at(i) = Eigen::MatrixXf(newHeight, newWidth);
    }

    for (int i=0; i<depth; i++) {
        Eigen::MatrixXf subMat = mat.at(i);
        for (int j=0; j<newHeight; j++) {
            for (int k=0; k<newWidth; k++) {
                returnMat.at(i)(j,k) = subMat.block(j*stride,k*stride,poolSize,poolSize).mean();
            }
        }
    }
    return returnMat;
}

std::vector<Eigen::MatrixXf> maxPool(const std::vector<Eigen::MatrixXf>& mat, int stride, int poolSize) {
    int depth = mat.size();
    int height = mat.at(0).rows();
    int width = mat.at(0).cols();

    int newHeight = height/stride;
    int newWidth = width/stride;

    std::vector<Eigen::MatrixXf> returnMat(depth);
    for (int i = 0; i < depth; ++i) {
        returnMat.at(i) = Eigen::MatrixXf(newHeight, newWidth);
    }

    for (int i=0; i<depth; i++) {
        Eigen::MatrixXf subMat = mat.at(i);
        for (int j=0; j<newHeight; j++) {
            for (int k=0; k<newWidth; k++) {
                returnMat.at(i)(j,k) = subMat.block(j*stride,k*stride,poolSize,poolSize).maxCoeff();
            }
        }
    }
    return returnMat;
}

std::vector<Eigen::MatrixXf> deriveAvgPool(const std::vector<Eigen::MatrixXf>& mat, const std::vector<Eigen::MatrixXf>& grad, int stride, int poolSize) {
    int depth = mat.size();
    int height = mat.at(0).rows();
    int width = mat.at(0).cols();

    std::vector<Eigen::MatrixXf> returnMat(depth);
    for (int i = 0; i < depth; ++i) {
        returnMat.at(i) = Eigen::MatrixXf(height, width);
    }

    for (int i = 0; i < depth; ++i) {
        const Eigen::MatrixXf& inputMat = mat.at(i);
        const Eigen::MatrixXf& gradMat = grad.at(i);

        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                float gradient = gradMat(j / stride, k / stride) / (poolSize * poolSize);

                for (int m = 0; m < poolSize; ++m) {
                    for (int n = 0; n < poolSize; ++n) {
                        int row = (j / stride) * stride + m;
                        int col = (k / stride) * stride + n;
                        if (row < height && col < width) {
                            returnMat.at(i)(row, col) += gradient;
                        }
                    }
                }
            }
        }
    }
    return returnMat;
}

std::vector<Eigen::MatrixXf> deriveMaxPool(const std::vector<Eigen::MatrixXf>& mat, const std::vector<Eigen::MatrixXf>& grad, int stride, int poolSize) {
    int depth = mat.size();
    int height = mat.at(0).rows();
    int width = mat.at(0).cols();

    std::vector<Eigen::MatrixXf> returnMat(depth);
    for (int i = 0; i < depth; ++i) {
        returnMat.at(i) = Eigen::MatrixXf(height, width);
        returnMat.at(i).setZero();
    }

    for (int i=0; i<depth; i++) {
        const Eigen::MatrixXf& subMat = mat.at(i);
        const Eigen::MatrixXf& subGrad = grad.at(i);
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                int startJ = (j/stride)*stride;
                int startK = (k/stride)*stride;

                Eigen::MatrixXf poolingRegion = subMat.block(startJ,startK,poolSize,poolSize);
                Eigen::Index row,col;
                poolingRegion.maxCoeff(&row,&col);
                
                if (startJ+row == j && startK+col == k) {
                    returnMat.at(i)(j,k) = subGrad(j/stride,j/stride);
                }
            }
        }
    }
    return returnMat;
}