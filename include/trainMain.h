#ifndef TRAIN_MAIN_H
#define TRAIN_MAIN_H

#include "Network.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <omp.h>

std::vector<std::string> collectFilePaths(std::string path);

void trainMain(std::string path, const Network& CNN, std::vector<int> inputMatDimensions);

std::vector<Eigen::MatrixXf> imgToMatrix(cv::Mat img, std::vector<int> inputMatDimensions);

#endif