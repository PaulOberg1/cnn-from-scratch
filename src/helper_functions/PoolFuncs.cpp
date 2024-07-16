#include "helper_functions/PoolFuncs.h"

const Eigen::MatrixXf& avgPool(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows/2;
    int newCols = cols/2;

    Eigen::MatrixXf returnMat (newRows,newCols);

    for (int i=0; i<cols; i++) {
        for (int j=0; j<rows; j++) {
            returnMat[i/2, j/2] = mat.block(i,j,2,2).mean();
        }
    }
    return returnMat;
}

const Eigen::MatrixXf& maxPool(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows/2;
    int newCols = cols/2;

    Eigen::MatrixXf returnMat (newRows,newCols);

    for (int i=0; i<cols; i++) {
        for (int j=0; j<rows; j++) {
            returnMat[i/2, j/2] = mat.block(i,j,2,2).maxCoeff();
        }
    }
    return returnMat;
}

const Eigen::MatrixXf& deriveAvgPool(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows*2;
    int newCols = cols*2;

    Eigen::MatrixXf returnMat(newRows,newCols);
    returnMat.setZero();

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            float gradValue = mat[i,j] / 4.0f;
            returnMat.block<2,2>(i*2,j*2).array()+=gradValue;
        }
    }
    return returnMat;
}

const Eigen::MatrixXf& deriveMaxPool(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    Eigen::MatrixXf returnMat (rows, cols);

    for (int i=0; i<cols; i++) {
        for (int j=0; j<rows; j++) {
            float maxVal = mat.block<2, 2>(i/2, j/2).maxCoeff();
            returnMat[i,j] = (mat[i,j] == maxVal) ? 1.0f : 0.0f;
        }
    }
    return returnMat;
}
