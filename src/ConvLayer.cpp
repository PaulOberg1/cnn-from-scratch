#include "ConvLayer.h"
#include <iostream>
#include <fstream>

ConvLayer::ConvLayer(std::vector<int> prevMatDimensions, std::vector<int> kernelDimensions, const ActivationFunc3D& activation, const ActivationFunc3DDeriv& activationDeriv, const PoolFunc& pool, const PoolFuncDeriv& poolDeriv, int poolStride, int poolSize)
    : m_activation(activation), m_activationDeriv(activationDeriv), m_pool(pool), m_poolDeriv(poolDeriv), m_poolStride(poolStride), m_poolSize(poolSize) {
        initWeights(prevMatDimensions,kernelDimensions);
}

void ConvLayer::initWeights(std::vector<int> prevMatDimensions, std::vector<int> kernelDimensions) {
    assert(prevMatDimensions.size() == 3 && kernelDimensions.size()==4);
    calcOutputSize(prevMatDimensions,kernelDimensions);

    const int numKernels = kernelDimensions.at(0);
    const int kDepth = kernelDimensions.at(1);
    const int kHeight = kernelDimensions.at(2);
    const int kWidth = kernelDimensions.at(3);

    float fanIn = getOutputSize().at(0);
    for (int i=0; i<numKernels; i++) {
        for (int j=0; j<kDepth; j++) {
            m_kernels.at(i).at(j) = ((Eigen::MatrixXf::Random(kHeight,kWidth).array()+1.0f)/2.0f) * std::sqrt(2.0f / fanIn);
        }
    }

    const int pHeight = prevMatDimensions.at(1);
    const int pWidth = prevMatDimensions.at(2);

    const int bHeight = pHeight + 1 - kHeight;
    const int bWidth = pWidth + 1 - kWidth;
    
    m_biases = Eigen::MatrixXf(bHeight,bWidth);
    m_biases.setZero();
    
}

void ConvLayer::forwardProp(std::vector<Eigen::MatrixXf> X) {
    try{
        m_Z = convolve(X, m_kernels, m_biases);
        m_A = m_activation(m_Z);
        m_P = m_pool(m_A, m_poolStride, m_poolSize);
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in ConvLayer::forwardProp : "<<e.what();
    }
}

std::vector<Eigen::MatrixXf> ConvLayer::reshapeTo3D(const Eigen::MatrixXf& oldMat, const std::vector<Eigen::MatrixXf>& newMat) {
    const int newDepth = newMat.size();
    const int newHeight = newMat.at(0).rows();
    const int newWidth = newMat.at(0).cols();

    assert(oldMat.size() == newDepth * newHeight * newWidth);

    std::vector<Eigen::MatrixXf> reshaped(newDepth, Eigen::MatrixXf(newHeight, newWidth));

    for (int d = 0; d < newDepth; ++d) {
        for (int h = 0; h < newHeight; ++h) {
            for (int w = 0; w < newWidth; ++w) {
                reshaped.at(d)(h, w) = oldMat(d * newHeight * newWidth + h * newWidth + w);
            }
        }
    }
    return reshaped;
}

void ConvLayer::backProp(Eigen::MatrixXf nextLayerW, Eigen::MatrixXf nextLayerDz, std::vector<Eigen::MatrixXf> layerInputMat) {
    try{
        Eigen::MatrixXf dF = nextLayerW * nextLayerDz;
        m_dP = reshapeTo3D(dF,m_P);
        
        m_dA = m_poolDeriv(m_A, m_dP, m_poolStride, m_poolSize);

        m_dZ = m_activationDeriv(m_Z,m_dA);
        
        for (int i=0; i<m_kernels.size(); i++) {
            for (int j=0; j<m_kernels.at(0).size(); j++) {
                m_dK.at(i).at(j) = convolve(m_P.at(j),m_dZ.at(i),0);
            }
        }
        
        for (int i=0; i<m_dZ.size(); i++) {
            m_dB(i) = m_dZ.at(i).sum();
        }
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in ConvLayer::backProp: "<<e.what();
    }
}

void ConvLayer::backProp(std::vector<std::vector<Eigen::MatrixXf>> nextLayerW, std::vector<Eigen::MatrixXf> nextLayerDz, std::vector<Eigen::MatrixXf> layerInputMat) {
    try{
        for (int i=0; i<m_dP.size(); i++)
            m_dP.at(i).setZero();
        for (int i=0; i<nextLayerW.size(); i++) {
            for (int j=0; j<nextLayerW.at(0).size(); j++) {
                for (int k=0; k<nextLayerW.at(0).at(0).rows(); k++) {
                    for (int l=0; l<m_dP.size(); l++) {
                        for (int m=0; m<nextLayerW.at(0).at(0).rows(); m++) {
                            for (int n=0; n<nextLayerW.at(0).at(0).cols(); n++) {
                                m_dP.at(l)(j+m,k+n) += nextLayerDz.at(i)(j,k) * nextLayerW.at(i).at(l)(m,n);
                            }
                        }
                    }
                }
            }
        }

        m_dA = m_poolDeriv(m_A, m_dP, m_poolStride, m_poolSize);

        m_dZ = m_activationDeriv(m_Z,m_dA);
        
        for (int i=0; i<m_kernels.size(); i++) {
            for (int j=0; j<m_kernels.at(0).size(); j++) {
                m_dK.at(i).at(j) = convolve(m_P.at(j),m_dZ.at(i),0);
            }
        }
        
        for (int i=0; i<m_dZ.size(); i++) {
            m_dB(i) = m_dZ.at(i).sum();
        }
    } catch (const std::exception& e) {
        std::cerr<<"Caught exception in ConvLayer::backProp: "<<e.what();
    }
}
    
void ConvLayer::gradDesc(double learningRate) {
    for (int i=0; i<m_kernels.size(); i++) {
        for (int j=0; j<m_kernels.size(); j++) {
            m_kernels.at(i).at(j) -= learningRate*m_dK.at(i).at(j);
        }
    }
    
    m_biases -= learningRate*m_dB;
}

std::vector<Eigen::MatrixXf> ConvLayer::getZ() {
    return m_Z;
}

std::vector<Eigen::MatrixXf> ConvLayer::getA() {
    return m_A;
}

std::vector<Eigen::MatrixXf> ConvLayer::getP() {
    return m_P;
}

std::vector<std::vector<Eigen::MatrixXf>> ConvLayer::getK() {
    return m_kernels;
}

std::vector<Eigen::MatrixXf> ConvLayer::getDz() {
    return m_dZ;
}

std::vector<int> ConvLayer::getOutputSize() {
    return m_outputSize;
}

Eigen::VectorXf ConvLayer::getFlattenedP() {
    
    std::vector<float> flattenedP = {};
    for (int i=0; i<m_P.size(); i++) {
        Eigen::MatrixXf subMat = m_P.at(i);
        flattenedP.insert(flattenedP.end(),subMat.data(),subMat.data()+subMat.size());
    }
    Eigen::VectorXf returnVec(flattenedP.size());
    std::copy(flattenedP.begin(),flattenedP.end(),returnVec.data());
    return returnVec;
}

void ConvLayer::calcOutputSize(std::vector<int> prevMatDimensions,std::vector<int> kernelDimensions) {
    const int pHeight = prevMatDimensions.at(1);
    const int pWidth = prevMatDimensions.at(2);

    const int newDepth = kernelDimensions.at(0);
    const int newHeight = (pHeight + 1 - kernelDimensions.at(2))/2;
    const int newWidth = (pWidth + 1 - kernelDimensions.at(3))/2;

    m_outputSize = {newDepth,newHeight,newWidth};
}

std::vector<Eigen::MatrixXf> ConvLayer::convolve(const std::vector<Eigen::MatrixXf>& mat, const std::vector<std::vector<Eigen::MatrixXf>>& kernels, const Eigen::VectorXf& biases) {
    const int numKernels = kernels.size();
    const int kDepth = kernels.at(0).size();
    const int kHeight = kernels.at(0).at(0).rows();
    const int kWidth = kernels.at(0).at(0).cols();

    const int mDepth = mat.size();
    const int mHeight = mat.at(0).rows();
    const int mWidth = mat.at(0).cols();

    const int outDepth = numKernels;
    const int outHeight = mHeight + 1 - kHeight;
    const int outWidth = mWidth + 1 - kWidth;

    std::vector<Eigen::MatrixXf> returnMat;
    for (int i=0; i<outDepth; i++) {
        returnMat.at(i) = Eigen::MatrixXf(outHeight,outWidth);
    }

    for (int i=0; i<outDepth; i++) {
        const std::vector<Eigen::MatrixXf>& kernel = kernels.at(i);
        for (int j=0; j<outHeight; j++) {
            for (int k=0; k<outWidth; k++) {
                int sum=0;
                for (int l=0; l<kDepth; l++) {
                    sum+=(kernel.at(l).block(j,k,j+kHeight,j+kWidth).array()).sum() + biases(j,k);
                }
                returnMat.at(i)(j,k) = sum;
            }
        }
    }
    return returnMat;
}

std::vector<Eigen::MatrixXf> ConvLayer::convolve(const std::vector<Eigen::MatrixXf>& mat, const std::vector<std::vector<Eigen::MatrixXf>>& kernels) {
    const int numKernels = kernels.size();
    const int kDepth = kernels.at(0).size();
    const int kHeight = kernels.at(0).at(0).rows();
    const int kWidth = kernels.at(0).at(0).cols();

    const int mDepth = mat.size();
    const int mHeight = mat.at(0).rows();
    const int mWidth = mat.at(0).cols();

    const int outDepth = numKernels;
    const int outHeight = mHeight + 1 - kHeight;
    const int outWidth = mWidth + 1 - kWidth;

    std::vector<Eigen::MatrixXf> returnMat;
    for (int i=0; i<outDepth; i++) {
        returnMat.at(i) = Eigen::MatrixXf(outHeight,outWidth);
    }

    for (int i=0; i<outDepth; i++) {
        const std::vector<Eigen::MatrixXf>& kernel = kernels.at(i);
        for (int j=0; j<outHeight; j++) {
            for (int k=0; k<outWidth; k++) {
                int sum=0;
                for (int l=0; l<kDepth; l++) {
                    sum+=(kernel.at(l).block(j,k,j+kHeight,j+kWidth).array()).sum();
                }
                returnMat.at(i)(j,k) = sum;
            }
        }
    }
    return returnMat;
}


Eigen::MatrixXf ConvLayer::convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, int padding) {

    int rows = inputMat.rows()+2*padding;
    int cols = inputMat.cols()+2*padding;
    assert(rows>0 && cols>0);

    int kRows = grad.rows();
    int kCols = grad.cols();
    assert(kRows>0 && kCols>0);

    int newRows = 1+rows-kRows;
    int newCols = 1+cols-kCols;
    assert(newRows>0 && newCols>0);

    Eigen::MatrixXf mat (rows, cols);
    mat.setZero();
    mat.block(padding,padding,rows-2*padding,cols-2*padding) = inputMat;

    Eigen::MatrixXf returnMat (newRows, newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            assert(i+kRows<=rows && j+kCols<=cols);
            returnMat(i, j) = (mat.block(i,j,kRows,kCols).array() * grad.array()).sum();
        }
    }
    return returnMat;
}

Eigen::MatrixXf ConvLayer::convolve(const Eigen::MatrixXf& inputMat, const Eigen::MatrixXf& grad, const Eigen::MatrixXf biases, int padding) {
    int rows = inputMat.rows()+2*padding;
    int cols = inputMat.cols()+2*padding;
    assert(rows>0 && cols>0);

    int kRows = grad.rows();
    int kCols = grad.cols();
    assert(kRows>0 && kCols>0);

    int newRows = biases.rows();
    int newCols = biases.cols();
    assert(newRows>0 && newCols>0);

    assert(biases.rows()==1+rows-kRows && biases.cols()==1+cols-kCols);

    Eigen::MatrixXf mat(rows,cols);
    mat.setZero();
    mat.block(padding,padding,rows-2*padding,cols-2*padding) = inputMat;

    Eigen::MatrixXf returnMat (newRows, newCols);

    for (int i=0; i<newRows; i++) {
        for (int j=0; j<newCols; j++) {
            assert(i+kRows<=rows && j+kCols<=cols);
            returnMat(i, j) = (mat.block(i,j,kRows,kCols).array() * grad.array()).sum() + biases(i, j);
        }
    }
    return returnMat;
}

/*
void ConvLayer::storeData(std::string path) {
    std::ofstream file(path,std::ios::binary);

    if (file.is_open()) {
        int k_rows = m_kernels.rows();
        int k_cols = m_kernels.cols();

        file.write(reinterpret_cast<char*>(&k_rows),sizeof(int));
        file.write(reinterpret_cast<char*>(&k_cols),sizeof(int));
        file.write(reinterpret_cast<const char*>(m_kernels.data()),k_rows*k_cols*sizeof(float));


        int b_rows = m_biases.rows();
        int b_cols = m_biases.cols();

        file.write(reinterpret_cast<char*>(&b_rows),sizeof(int));
        file.write(reinterpret_cast<char*>(&b_cols),sizeof(int));
        file.write(reinterpret_cast<const char*>(m_biases.data()),b_rows*b_cols*sizeof(float));

        file.close();
    }
}
*/