#pragma once
#include <QObject>
#include <QImage>
#include <QTimer>
#include <array>
#include <memory>
#include <chrono>
#include <deque>
#include "app_settings.hpp"
#include "video_capture.hpp"
#include "zmq_publisher.hpp"
#include "zmq_subscriber.hpp"
#include "image_processor.hpp"
#include <QThread>
class SystemManager : public QObject {
    Q_OBJECT
public:
    explicit SystemManager(QObject* parent = nullptr);
    ~SystemManager() override;

    bool start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const { return is_monitoring_; }
    bool is_backend_online() const { return is_backend_online_; }

    // Áp dụng cài đặt: phòng/giám thị/tốc độ gửi có hiệu lực ngay, camera có hiệu lực ở lần BẮT ĐẦU sau
    void apply_settings(const AppSettings& settings);
    const AppSettings& settings() const { return settings_; }
    void request_seat_relearn();

    double get_current_fps();
    uint64_t get_frame_count() const { return frame_count_; }
    // Kích thước khung hình camera THỰC SỰ trả về (0x0 khi chưa có frame)
    int get_frame_width() const { return frame_w_; }
    int get_frame_height() const { return frame_h_; }
    // Định dạng camera trả về ("MJPG" là đúng; "YUY2" ở 1080p chỉ ~5-10 fps qua USB 2.0)
    QString get_frame_format() const {
        return video_capture_ ? QString::fromStdString(video_capture_->fourcc_string()) : QString();
    }
    uint64_t get_ai_count() const;
    // Provide access to image processor for UI connections
    ImageProcessorWorker* get_image_processor() const { return image_processor_; }

signals:
    void backendStatusChanged(bool is_online);
    void processResultRequested(const exam::InferenceResult& result);
    // Độ trễ từ lúc chụp frame tới lúc nhận kết quả AI (ms) và số kết quả AI/giây
    void aiMetricsUpdated(double latency_ms, double ai_hz);

private slots:
    void on_frame_captured(const cv::Mat& frame, int frame_id);
    void on_result_received(const exam::InferenceResult& result);
    void check_heartbeat();

private:
    AppSettings settings_;
    std::unique_ptr<VideoCapture> video_capture_;
    std::unique_ptr<ZmqPublisher> zmq_publisher_;
    std::unique_ptr<ZmqSubscriber> zmq_subscriber_;

    ImageProcessorWorker* image_processor_ = nullptr;
    QThread* processor_thread_ = nullptr;

    QTimer* heartbeat_timer_;
    bool is_monitoring_ = false;
    bool is_backend_online_ = false;

    // Stats
    uint64_t frame_count_ = 0;
    int frame_w_ = 0;
    int frame_h_ = 0;
    uint64_t last_frame_count_ = 0;
    double current_fps_ = 0.0;
    std::chrono::steady_clock::time_point last_fps_time_;
    int last_processed_frame_id_ = -1;

    // Đo độ trễ: thời điểm chụp của 512 frame gần nhất (theo frame_id)
    std::array<std::chrono::steady_clock::time_point, 512> capture_time_{};
    std::deque<std::chrono::steady_clock::time_point> result_times_;
    double latency_ema_ms_ = 0.0;
};
