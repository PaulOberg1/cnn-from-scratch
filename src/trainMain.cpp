#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


Eigen::MatrixXf imgToMatrix(cv::Mat img, int matSideLength) {
    cv::Mat resizedImg;
    cv::resize(img,resizedImg,cv::Size(matSideLength,matSideLength));

    cv::Mat grayImg;
    cv::cvtColor(resizedImg,grayImg,cv::COLOR_BGR2GRAY);

    cv::Mat normalisedImg;
    grayImg.convertTo(normalisedImg, CV_32F, 1.0 / 255.0);

    Eigen::MatrixXf X (matSideLength,matSideLength);
    for (int i=0; i<matSideLength; i++) {
        for (int j=0; j<matSideLength; j++) {
            X(i,j) = normalisedImg.at<float>(i,j);
        }
    }
}