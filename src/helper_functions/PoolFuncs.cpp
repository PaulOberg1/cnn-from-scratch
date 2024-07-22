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

Eigen::MatrixXf deriveAvgPool(const Eigen::MatrixXf& mat, int stride, int poolSize) {
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

Eigen::MatrixXf deriveMaxPool(const Eigen::MatrixXf& mat, int stride, int poolSize) {
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
