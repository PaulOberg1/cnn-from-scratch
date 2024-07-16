#include "helper_functions/PoolFuncs.h"

const Eigen::MatrixXf& avgPool(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    int newRows = rows/2;
    int newCols = cols/2;

    Eigen::MatrixXf returnMat (newCols,newRows);

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

    Eigen::MatrixXf returnMat (newCols,newRows);

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

    Eigen::MatrixXf returnMat(newCols,newRows);

    for (int i=0; i<newCols; i++) {
        for (int j=0; j<newRows; j++) {
            returnMat[i,j] = mat[i/2,j/2] / 4;
        }
    }
    return returnMat;
}

const Eigen::MatrixXf& deriveMaxPool(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();

    Eigen::MatrixXf returnMat (cols, rows);

    for (int i=0; i<cols; i++) {
        for (int j=0; j<rows; j++) {
            float maxVal = mat.block(i/2,j/2,2,2).maxCoeff();
            returnMat[i,j] = mat[i,j] == maxVal;
        }
    }
    return returnMat;
}
