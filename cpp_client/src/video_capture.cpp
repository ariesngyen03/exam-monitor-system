#include "video_capture.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

VideoCapture::VideoCapture(int camera_index, const std::string& source, QObject* parent)
    : QThread(parent), camera_index_(camera_index), source_(source), running_(false), frame_id_(0) {}

VideoCapture::~VideoCapture() {
    stop();
}

bool VideoCapture::open_source() {
    if (!source_.empty()) {
        cap_.open(source_, cv::CAP_ANY);
        if (!cap_.isOpened()) {
            std::cerr << "[VideoCapture] Khong the mo nguon video: " << source_ << std::endl;
            return false;
        }
        if (is_file_) {
            double fps = cap_.get(cv::CAP_PROP_FPS);
            file_fps_ = (fps > 1.0 && fps < 120.0) ? fps : 30.0;
        }
        return true;
    }

    cap_.open(camera_index_, cv::CAP_ANY);
    if (!cap_.isOpened()) {
        // Một số webcam USB trên Windows chỉ mở được qua DirectShow
        cap_.open(camera_index_, cv::CAP_DSHOW);
    }
    if (!cap_.isOpened()) {
        std::cerr << "[VideoCapture] Khong the mo camera index: " << camera_index_ << std::endl;
        return false;
    }
    // THỨ TỰ BẮT BUỘC với backend DirectShow của OpenCV 4.12 (bản build không có MSMF): FPS -> rộng -> cao -> FOURCC.
    // Đặt FOURCC trước (như bản cũ) thì OpenCV áp MJPG cho kích thước hiện tại rồi xoá yêu cầu MJPG; lúc đặt 1920x1080
    // nó tự chọn YUY2 (đứng trước MJPG trong danh sách ưu tiên) — YUY2 1080p không nén qua USB 2.0 chỉ ~5-10 fps
    // (máy chạy thử 15/10/2026: camera 9 fps). Đặt FOURCC sau cùng thì lần cấu hình cuối là MJPG 1920x1080 @ 30 fps.
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap_.set(cv::CAP_PROP_FPS, 30);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, req_width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, req_height_);
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    fourcc_.store(static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC)));
    // Webcam có thể âm thầm trả độ phân giải/định dạng khác yêu cầu -> ghi lại giá trị thật (cũng hiện dưới khung video)
    std::cout << "[VideoCapture] Camera " << camera_index_ << " (" << cap_.getBackendName() << "): yeu cau "
              << req_width_ << "x" << req_height_ << " MJPG, camera tra " << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << " " << fourcc_string() << " @ " << cap_.get(cv::CAP_PROP_FPS)
              << " fps" << std::endl;
    return true;
}

std::string VideoCapture::fourcc_string() const {
    const int f = fourcc_.load();
    if (f <= 0) return {};
    std::string s;
    for (int i = 0; i < 4; ++i) {
        const char c = static_cast<char>((f >> (8 * i)) & 0xFF);
        s += (c >= 32 && c < 127) ? c : '?';
    }
    return s;
}

bool VideoCapture::start_capture() {
    if (running_.load()) return true;

    std::error_code ec;
    is_file_ = !source_.empty() && std::filesystem::is_regular_file(std::filesystem::u8path(source_), ec);
    if (!open_source()) return false;

    if (is_file_) {
        running_.store(true);
        frame_id_ = 0;
        start();
        std::cout << "[VideoCapture] Phat file video: " << source_ << " (" << file_fps_ << " fps)" << std::endl;
        return true;
    }

    running_.store(true);
    frame_id_ = 0;

    // Ưu tiên cao: luồng đọc camera không được nhường CPU cho AI (Python chạy cùng máy) — mất nhịp là video giật
    start(QThread::HighPriority);
    std::cout << "[VideoCapture] Camera " << camera_index_ << " da khoi dong" << std::endl;
    return true;
}

void VideoCapture::stop() {
    running_.store(false);
    if (isRunning()) {
        wait();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
}

void VideoCapture::run() {
    cv::Mat frame;
    int empty_reads = 0;
    const auto frame_period = std::chrono::microseconds(static_cast<long long>(1e6 / file_fps_));
    auto next_due = std::chrono::steady_clock::now();
    auto last_frame = std::chrono::steady_clock::now();
    while (running_.load()) {
        if (is_file_) {
            // Phát file đúng tốc độ gốc (như camera thật), hết file thì quay lại đầu
            std::this_thread::sleep_until(next_due);
            next_due += frame_period;
        }
        cap_ >> frame;
        if (frame.empty() && is_file_) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap_ >> frame;
        }
        if (frame.empty()) {
            if (++empty_reads % 500 == 0)
                std::cerr << "[VideoCapture] Camera khong tra frame..." << std::endl;
            // Camera USB lỏng cáp / camera IP rớt mạng: quá 3 giây không có frame thì mở lại nguồn
            if (!is_file_ && std::chrono::steady_clock::now() - last_frame > std::chrono::seconds(3)) {
                std::cerr << "[VideoCapture] Mat tin hieu camera, dang ket noi lai..." << std::endl;
                cap_.release();
                for (int i = 0; i < 20 && running_.load(); ++i) msleep(100);
                if (running_.load() && open_source())
                    std::cerr << "[VideoCapture] Da mo lai nguon video" << std::endl;
                last_frame = std::chrono::steady_clock::now();
            }
            msleep(2);
            continue;
        }
        empty_reads = 0;
        last_frame = std::chrono::steady_clock::now();
        // cap_ tái sử dụng bộ nhớ của `frame` -> clone 1 lần, các slot nhận chia sẻ chung bản sao này
        emit frameReady(frame.clone(), ++frame_id_);
    }
}
