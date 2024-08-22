#include "trainMain.h"

void trainMain(std::string path, const Network& CNN) {
    
    int yVal = 0;

    std::vector<std::vector<std::string>> paths;
    std::vector<std::string> damagedPaths = collectFilePaths("C:/EggSpector/data/Damaged");
    std::vector<std::string> unDamagedPaths = collectFilePaths("C:/EggSpector/data/Not Damaged");
    paths.push_back(damagedPaths);
    paths.push_back(unDamagedPaths);



    for (const auto& pathArr : paths) {

        #pragma omp parallel for
        for (int j=0; j<pathArr.size(); j++) {
            std::string imagePath = pathArr.at(j);
            cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
            Eigen::MatrixXf X = imgToMatrix(img,66);
            Eigen::MatrixXf Y (1,1);
            Y << yVal;

            
            Eigen::MatrixXf y_pred = CNN.run(100,0.01,X,Y);
            std::cout<<y_pred;
            
        }
    }
}

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
    return X;
}