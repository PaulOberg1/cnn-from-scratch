#include "helper_functions/PoolFuncs.h"

Eigen::MatrixXf avgPool(const Eigen::MatrixXf& mat, int stride, int poolSize) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows/stride;
    int newCols = cols/stride;

    Eigen::MatrixXf returnMat (newRows,newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            returnMat(i, j) = mat.block(i*stride,j*stride,poolSize,poolSize).mean();
        }
    }
    return returnMat;
}
std::vector<Eigen::MatrixXf> avgPool(const std::vector<Eigen::MatrixXf>& mat, int stride=2, int poolSize=2) {
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


Eigen::MatrixXf maxPool(const Eigen::MatrixXf& mat, int stride, int poolSize) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows/2;
    int newCols = cols/2;

    Eigen::MatrixXf returnMat (newRows,newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            returnMat(i, j) = mat.block(i*stride,j*stride,poolSize,poolSize).maxCoeff();
        }
    }
    return returnMat;
}
std::vector<Eigen::MatrixXf> maxPool(const std::vector<Eigen::MatrixXf>& mat, int stride=2, int poolSize=2) {
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

Eigen::MatrixXf deriveAvgPool(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad, int stride, int poolSize) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows*stride;
    int newCols = cols*stride;

    Eigen::MatrixXf returnMat(newRows,newCols);
    returnMat.setZero();

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            float gradValue = mat(i,j) / float(poolSize*poolSize);
            returnMat.block(i*stride, j*stride, poolSize, poolSize).array()+=gradValue;
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

Eigen::MatrixXf deriveMaxPool(const Eigen::MatrixXf& mat, const Eigen::MatrixXf& grad, int stride, int poolSize) {
    int rows = mat.rows();
    int cols = mat.cols();

    Eigen::MatrixXf returnMat (rows, cols);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            float maxVal = mat.block(i/stride, j/stride, poolSize, poolSize).maxCoeff();
            returnMat(i,j) = (mat(i,j) == maxVal) ? 1.0f : 0.0f;
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