#ifndef TRAIN_MAIN_H
#define TRAIN_MAIN_H

#include "Network.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <filesystem>

void trainMain(std::string path, const Network& CNN);

Eigen::MatrixXf imgToMatrix(cv::Mat img, int matSideLength);

#endif