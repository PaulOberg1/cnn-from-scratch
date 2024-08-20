#include "testMain.h"

double testImg(std::string imgPath, int matSideLength, const Network& CNN) {
    cv::Mat img = cv::imread(imgPath);
    Eigen::MatrixXf mat = imgToMatrix(img,matSideLength);
    Eigen::MatrixXf Y = CNN.forwardProp(mat);
    return Y(0,0);
}