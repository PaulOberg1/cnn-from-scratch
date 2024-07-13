#include <Eigen/Dense>

class DenseLayer{
    private:

    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;

    public:
    DenseLayer(int prevLayerNodes, int curLayerNodes) {
        initWeights(prevLayerNodes,curLayerNodes);
    }

    void initWeights(int prevLayerNodes, int curLayerNodes) {
        weights = Eigen::MatrixXf::Random(curLayerNodes,prevLayerNodes);
        biases = Eigen::MatrixXf::Random(curLayerNodes);
    }
};