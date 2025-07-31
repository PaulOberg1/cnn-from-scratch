# CNN Server MVP

A minimal C++ project demonstrating how to serve a custom CNN model with:

- A POSIX asynchronous HTTP server with a pthread thread pool  
- A ZeroMQ-based in-memory queue for dispatching image classification requests  
- A worker process that consumes requests, runs CNN inference, and logs results  
- Designed for horizontal scaling and easy integration with a load balancer (e.g., NGINX)

---

## Features

- Minimal HTTP GET `/predict?img=/path/to/image.jpg` endpoint  
- Basic HTTP parsing and response  
- Thread pool for concurrent request handling  
- ZeroMQ PUSH/PULL queue for decoupled request dispatch  
- Simple file path validation  