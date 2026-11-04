#include "zmq_publisher.hpp"
#include <iostream>
#include <chrono>

ZmqPublisher::ZmqPublisher(const std::string& address, QObject* parent)
    : QThread(parent), address_(address), running_(false), sent_count_(0) {
}

ZmqPublisher::~ZmqPublisher() {
    stop();
}

void ZmqPublisher::start_publish() {
    if (running_.load()) return;
    running_.store(true);
    start();
}

void ZmqPublisher::stop() {
    running_.store(false);
    queue_cond_.wakeOne();
    if (isRunning()) {
        wait();
    }
}

void ZmqPublisher::set_max_fps(int fps) {
    min_interval_ms_.store(fps > 0 ? 1000 / fps : 0);
}

void ZmqPublisher::set_session_info(const std::string& room, const std::string& proctor) {
    QMutexLocker lock(&queue_mutex_);
    room_ = room;
    proctor_ = proctor;
}

void ZmqPublisher::set_ai_config(const std::string& sensitivity, int absent_alert_seconds, const std::string& seatmap_path) {
    QMutexLocker lock(&queue_mutex_);
    sensitivity_ = sensitivity.empty() ? "medium" : sensitivity;
    absent_alert_seconds_ = absent_alert_seconds > 0 ? absent_alert_seconds : 15;
    seatmap_path_ = seatmap_path;
}

void ZmqPublisher::send_frame(const cv::Mat& frame, int frame_id) {
    if (!running_.load()) return;

    // Bỏ qua frame đến quá dày so với giới hạn — tránh nén JPEG vô ích
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_accept_).count();
    if (elapsed < min_interval_ms_.load()) return;
    last_accept_ = now;

    QMutexLocker lock(&queue_mutex_);
    pending_frame_ = frame;   // cv::Mat chia sẻ dữ liệu; VideoCapture đã clone mỗi frame
    pending_frame_id_ = frame_id;
    has_pending_ = true;
    queue_cond_.wakeOne();
}

void ZmqPublisher::run() {
    zmq::context_t context(1);
    zmq::socket_t publisher(context, zmq::socket_type::pub);
    publisher.set(zmq::sockopt::sndhwm, 2);
    publisher.set(zmq::sockopt::linger, 0);
    publisher.bind(address_);
    std::cout << "[ZmqPublisher] Bind thanh cong tai: " << address_ << std::endl;

    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
    std::vector<uchar> buffer;
    std::string serialized;

    while (running_.load()) {
        cv::Mat frame;
        int frame_id = 0;
        exam::FramePacket packet;

        {
            QMutexLocker lock(&queue_mutex_);
            while (!has_pending_ && running_.load()) {
                queue_cond_.wait(&queue_mutex_);
            }
            if (!running_.load()) break;

            frame = pending_frame_;
            frame_id = pending_frame_id_;
            pending_frame_.release();
            has_pending_ = false;
            packet.set_exam_room(room_);
            packet.set_proctor(proctor_);
            packet.set_sensitivity(sensitivity_);
            packet.set_absent_alert_seconds(absent_alert_seconds_);
            packet.set_seatmap_path(seatmap_path_);
        }

        if (frame.empty()) continue;

        cv::imencode(".jpg", frame, buffer, params);

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        packet.set_timestamp(ms);
        packet.set_frame_id(frame_id);
        packet.set_image_data(buffer.data(), buffer.size());
        packet.set_width(frame.cols);
        packet.set_height(frame.rows);
        packet.set_reset_seq(reset_seq_.load());

        packet.SerializeToString(&serialized);
        publisher.send(zmq::buffer(serialized), zmq::send_flags::dontwait);
        sent_count_++;
    }
    publisher.close();
    context.close();
    std::cout << "[ZmqPublisher] Da dong ket noi." << std::endl;
}
