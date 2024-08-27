#include "testMain.h"

double testImg(std::string imgPath, const Network& CNN, std::vector<int> inputMatDimensions) {
    cv::Mat img = cv::imread(imgPath);
    std::vector<Eigen::MatrixXf> mat = imgToMatrix(img,inputMatDimensions);
    Eigen::MatrixXf Y = CNN.forwardProp(mat);
    return Y(0,0);
}