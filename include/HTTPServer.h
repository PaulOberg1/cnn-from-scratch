#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include <string>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

void enqueue_request(int client_fd);
int dequeue_request();
std::string parse_http_request(int client_fd);
bool is_valid_path(const std::string& path);
void send_http_response(int client_fd, const std::string& body);
void* worker_thread(void* arg);
void start_http_server();

#endif