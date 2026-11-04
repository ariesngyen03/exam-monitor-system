#pragma once

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include "exam_monitor.pb.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class ZmqPublisher : public QThread {
    Q_OBJECT

public:
    explicit ZmqPublisher(const std::string& address = "tcp://*:5555", QObject* parent = nullptr);
    ~ZmqPublisher() override;

    void start_publish();
    void stop();

    // Giới hạn số frame/giây gửi sang AI (AI chỉ phân tích 2 nhịp/giây — gửi 30 fps là lãng phí CPU nén JPEG)
    void set_max_fps(int fps);
    void set_session_info(const std::string& room, const std::string& proctor);
    void set_ai_config(const std::string& sensitivity, int absent_alert_seconds, const std::string& seatmap_path);
    void request_seat_relearn() { reset_seq_.fetch_add(1); }

public slots:
    void send_frame(const cv::Mat& frame, int frame_id);

protected:
    void run() override;

private:
    std::string address_;
    std::atomic<bool> running_;
    uint64_t sent_count_;

    QMutex queue_mutex_;
    QWaitCondition queue_cond_;
    // Chỉ giữ 1 frame mới nhất chờ gửi (AI cũng chỉ dùng frame mới nhất)
    cv::Mat pending_frame_;
    int pending_frame_id_ = 0;
    bool has_pending_ = false;

    std::atomic<int> min_interval_ms_{250};
    std::chrono::steady_clock::time_point last_accept_{};
    std::string room_;
    std::string proctor_;
    std::string sensitivity_{"medium"};
    int absent_alert_seconds_ = 15;
    std::string seatmap_path_;
    std::atomic<int> reset_seq_{0};
};
