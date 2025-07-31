#include "ZMQWorker.h"

// ==== ZeroMQ Worker ====
void start_zmq_worker() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_PULL);
    socket.bind("tcp://*:5555");

    while (true) {
        zmq::message_t msg;
        socket.recv(msg, zmq::recv_flags::none);
        std::string image_path(static_cast<char*>(msg.data()), msg.size());

        std::string result = "Mock result for " + image_path; // replace with classify_image(image_path);
        std::cout << "Processed: " << image_path << " => " << result << std::endl;
    }
}