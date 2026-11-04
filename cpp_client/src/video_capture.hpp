#pragma once

#include <QThread>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <string>
#include <iostream>

class VideoCapture : public QThread {
    Q_OBJECT

public:
    // source rỗng -> mở camera theo chỉ số; ngược lại là đường dẫn file video / URL rtsp:// http://
    explicit VideoCapture(int camera_index = 0, const std::string& source = {}, QObject* parent = nullptr);
    ~VideoCapture() override;

    // Độ phân giải yêu cầu camera (gọi trước start_capture; không áp dụng cho file video / URL)
    void set_resolution(int width, int height) { req_width_ = width; req_height_ = height; }
    bool start_capture();
    void stop();
    // Định dạng camera thật sự trả về ("MJPG", "YUY2"...; rỗng với file video / URL) — gọi được từ mọi luồng
    std::string fourcc_string() const;

signals:
    // Tín hiệu phát ra ngay khi đọc xong frame (mỗi frame là 1 bản sao riêng, an toàn chia sẻ giữa các luồng)
    void frameReady(const cv::Mat& frame, int frame_id);

protected:
    void run() override;

private:
    bool open_source();   // mở camera / file / URL (dùng cả lúc khởi động và khi kết nối lại)

    int camera_index_;
    std::string source_;
    bool is_file_ = false;   // file video: phát theo đúng fps gốc và lặp lại khi hết
    double file_fps_ = 30.0;
    int req_width_ = 1920;
    int req_height_ = 1080;
    cv::VideoCapture cap_;
    std::atomic<int> fourcc_{0};
    std::atomic<bool> running_;
    int frame_id_;
};
