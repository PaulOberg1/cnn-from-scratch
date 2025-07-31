#include "HTTPServer.h"


constexpr int PORT = 8081;
constexpr int THREAD_POOL_SIZE = 4;

std::queue<int> request_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;

void enqueue_request(int client_fd) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.notify_one();
}

int dequeue_request() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [] { return !request_queue.empty(); });
    int fd = request_queue.front();
    request_queue.pop();
    return fd;
}

std::string parse_http_request(int client_fd) {
    char buffer[2048] = {0};
    recv(client_fd, buffer, sizeof(buffer), 0);
    std::string request(buffer);

    size_t pos = request.find("GET /predict?img=");
    if (pos == std::string::npos) return "";
    size_t start = pos + 17; // length of "/predict?img="
    size_t end = request.find(" ", start);
    return request.substr(start, end - start);
}

bool is_valid_path(const std::string& path) {
    return access(path.c_str(), R_OK) == 0;
}

void send_http_response(int client_fd, const std::string& body) {
    std::string response =
        "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
        "\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n" + body;
    send(client_fd, response.c_str(), response.size(), 0);
}

void* worker_thread(void* arg) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_PUSH);
    socket.connect("tcp://localhost:5555");

    while (true) {
        int client_fd = dequeue_request();
        std::string image_path = parse_http_request(client_fd);

        if (!is_valid_path(image_path)) {
            send_http_response(client_fd, "Invalid image path");
            close(client_fd);
            continue;
        }

        socket.send(zmq::buffer(image_path), zmq::send_flags::none);
        send_http_response(client_fd, "Request queued: " + image_path);
        close(client_fd);
    }
    return nullptr;
}

void start_http_server() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 10);

    for (int i = 0; i < THREAD_POOL_SIZE; ++i) {
        pthread_t thread;
        pthread_create(&thread, nullptr, worker_thread, nullptr);
        pthread_detach(thread);
    }

    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        enqueue_request(client_fd);
    }
}


// ==== Main ====
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "worker") {
        start_zmq_worker();
    } else {
        start_http_server();
    }
    return 0;
}
