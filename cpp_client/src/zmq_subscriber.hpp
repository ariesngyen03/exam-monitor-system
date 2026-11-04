#pragma once

#include <QThread>
#include <zmq.hpp>
#include "exam_monitor.pb.h"
#include <atomic>
#include <iostream>

class ZmqSubscriber : public QThread {
    Q_OBJECT

public:
    explicit ZmqSubscriber(const std::string& address = "tcp://localhost:5556", QObject* parent = nullptr);
    ~ZmqSubscriber() override;

    void start_listen();
    void stop();

    uint64_t get_received_count() const;

signals:
    // Phát tín hiệu khi parse xong InferenceResult
    void resultReady(const exam::InferenceResult& result);

protected:
    void run() override;

private:
    std::string address_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> received_count_;
};
