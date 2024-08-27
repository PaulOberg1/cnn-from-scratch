#include "trainMain.h"
#include <random>

std::vector<std::string> collectFilePaths(std::string path) {
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::string imagePath = entry.path().string();
        paths.push_back(imagePath);
    }
    return paths;
}

void trainMain(std::string path, const Network& CNN, std::vector<int> inputMatDimensions) {
    
    
    std::vector<std::string> damagedPaths = collectFilePaths("C:/EggSpector/data/Damaged");
    std::vector<std::string> unDamagedPaths = collectFilePaths("C:/EggSpector/data/Not Damaged");

    damagedPaths = std::vector(damagedPaths.begin(),damagedPaths.begin()+unDamagedPaths.size());

    std::vector<std::pair<std::string,int>> damagedArr(damagedPaths.size());
    std::transform(damagedPaths.begin(),damagedPaths.end(),damagedArr.begin(),[](std::string path) {return std::make_pair(path,1);});

    std::vector<std::pair<std::string,int>> unDamagedArr(unDamagedPaths.size());
    std::transform(unDamagedPaths.begin(),unDamagedPaths.end(),unDamagedArr.begin(),[](std::string path) {return std::make_pair(path,0);});

    std::vector<std::pair<std::string,int>> paths;
    paths.insert(paths.end(),damagedArr.begin(),damagedArr.end());
    paths.insert(paths.end(),unDamagedArr.begin(),unDamagedArr.end());
    

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(paths.begin(), paths.end(), g);

    
    int failCount=0;
    int winCount=0;
    #pragma omp parallel for
    for (int j=0; j<paths.size(); j++) {
        const auto& pair = paths.at(j);
        std::string imagePath = pair.first;
        int val = pair.second;
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        std::vector<Eigen::MatrixXf> X = imgToMatrix(img,inputMatDimensions);
        if (j%10) {
            Eigen::MatrixXf y_pred = CNN.forwardProp(X);
            float margin = abs(y_pred(0,0)-val); 
            if (margin>0.5) 
                failCount++;
            
            else
                winCount++;
            std::string precision = std::to_string(int((1-margin)*100))+"%";
            std::cout<<precision<<"\t"<<y_pred<<"\t"<<val;
            std::cout<<"\tcurPrec: "<<float(winCount)/float(failCount+winCount);
        }

        Eigen::MatrixXf Y (1,1);
        Y << val;

        Eigen::MatrixXf y_pred = CNN.run(100,0.01,X,Y);
    }
}

std::vector<Eigen::MatrixXf> imgToMatrix(cv::Mat img, std::vector<int> inputMatDimensions) {
    int inputDepth = inputMatDimensions.at(0);
    int inputHeight = inputMatDimensions.at(1);
    int inputWidth = inputMatDimensions.at(2);

    cv::Mat resizedImg;
    cv::resize(img,resizedImg,cv::Size(inputHeight,inputWidth));

    cv::Mat normalisedImg;
    resizedImg.convertTo(normalisedImg, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(inputDepth);
    cv::split(normalisedImg,channels);

    std::vector<Eigen::MatrixXf> X (inputDepth);
    for (int i=0; i<inputDepth; i++) {
        const cv::Mat& channel = channels.at(i);
        for (int j=0; j<inputHeight; j++) {
            for (int k=0; j<inputWidth; k++) {
                X.at(i)(j,k) = channel.at<float>(j,k);
            }
        }
    }
    return X;
}