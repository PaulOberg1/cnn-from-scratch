#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include "Network.h"
#include "trainMain.h"

#include <opencv2/opencv.hpp>

double testImg(std::string imgPath, const Network& CNN, std::vector<int> inputMatDimensions);

#endif