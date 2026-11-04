#include "zmq_subscriber.hpp"

ZmqSubscriber::ZmqSubscriber(const std::string& address, QObject* parent)
    : QThread(parent), address_(address), running_(false), received_count_(0) {}

ZmqSubscriber::~ZmqSubscriber() {
    stop();
}

void ZmqSubscriber::start_listen() {
    if (running_.load()) return;
    running_.store(true);
    start();
}

void ZmqSubscriber::stop() {
    running_.store(false);
    if (isRunning()) {
        wait();
    }
}

uint64_t ZmqSubscriber::get_received_count() const {
    return received_count_.load();
}

void ZmqSubscriber::run() {
    // Socket ZMQ nên được khởi tạo bên trong Thread sở hữu nó
    zmq::context_t context(1);
    zmq::socket_t subscriber(context, zmq::socket_type::sub);
    
    subscriber.set(zmq::sockopt::subscribe, "");
    subscriber.set(zmq::sockopt::conflate, 1);
    subscriber.set(zmq::sockopt::rcvtimeo, 100); 
    subscriber.connect(address_);
    
    std::cout << "[ZmqSubscriber] Ket noi va lang nghe tai: " << address_ << std::endl;

    while (running_.load()) {
        zmq::message_t message;
        auto recv_res = subscriber.recv(message, zmq::recv_flags::none);
        
        if (!recv_res.has_value()) continue;

        exam::InferenceResult inference_result;
        if (inference_result.ParseFromArray(message.data(), static_cast<int>(message.size()))) {
            received_count_++;
            emit resultReady(inference_result);
        }
    }
    
    subscriber.close();
    context.close();
    std::cout << "[ZmqSubscriber] Da ngat ket noi." << std::endl;
}
