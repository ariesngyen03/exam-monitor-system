#include "system_manager.hpp"
#include <iostream>

SystemManager::SystemManager(QObject *parent) : QObject(parent) {
  qRegisterMetaType<std::vector<exam::Detection>>(
      "std::vector<exam::Detection>");
  qRegisterMetaType<exam::InferenceResult>("exam::InferenceResult");

  settings_ = AppSettings::load();

  processor_thread_ = new QThread(this);
  image_processor_ = new ImageProcessorWorker();
  image_processor_->set_enable_emotion(settings_.enable_emotion);
  image_processor_->moveToThread(processor_thread_);
  connect(processor_thread_, &QThread::finished, image_processor_, &QObject::deleteLater);
  processor_thread_->start();

  connect(this, &SystemManager::processResultRequested,
          image_processor_, &ImageProcessorWorker::process_result,
          Qt::QueuedConnection);

  heartbeat_timer_ = new QTimer(this);
  heartbeat_timer_->setSingleShot(false);
  connect(heartbeat_timer_, &QTimer::timeout, this,
          &SystemManager::check_heartbeat);

  // Kết nối AI ngay khi mở app (độc lập với camera) để trạng thái Backend
  // phản ánh đúng AI server có đang chạy hay không. AI gửi heartbeat mỗi 1s.
  zmq_subscriber_ = std::make_unique<ZmqSubscriber>("tcp://127.0.0.1:5556");
  connect(zmq_subscriber_.get(), &ZmqSubscriber::resultReady, this,
          &SystemManager::on_result_received);
  zmq_subscriber_->start_listen();
  heartbeat_timer_->start(5000);
}

SystemManager::~SystemManager() {
  stop_monitoring();
  heartbeat_timer_->stop();
  if (zmq_subscriber_) {
    zmq_subscriber_->stop();
    zmq_subscriber_.reset();
  }
  if (processor_thread_) {
    processor_thread_->quit();
    processor_thread_->wait();
  }
}

void SystemManager::apply_settings(const AppSettings &settings) {
  settings_ = settings;
  settings_.save();
  if (zmq_publisher_) {
    zmq_publisher_->set_max_fps(settings_.send_fps);
    zmq_publisher_->set_session_info(settings_.exam_room.toStdString(),
                                     settings_.proctor.toStdString());
    zmq_publisher_->set_ai_config(settings_.sensitivity.toStdString(),
                                  settings_.absent_alert_seconds,
                                  settings_.seatmap_path.toStdString());
  }
  if (image_processor_) {
    image_processor_->set_enable_emotion(settings_.enable_emotion);
  }
}

void SystemManager::request_seat_relearn() {
  if (zmq_publisher_) zmq_publisher_->request_seat_relearn();
}

bool SystemManager::start_monitoring() {
  try {
    video_capture_ = std::make_unique<VideoCapture>(settings_.camera_index,
                                                    settings_.camera_source.toStdString());
    video_capture_->set_resolution(settings_.camera_height >= 1080 ? 1920 : 1280, settings_.camera_height);
    zmq_publisher_ = std::make_unique<ZmqPublisher>("tcp://*:5555");
    zmq_publisher_->set_max_fps(settings_.send_fps);
    zmq_publisher_->set_session_info(settings_.exam_room.toStdString(),
                                     settings_.proctor.toStdString());
    zmq_publisher_->set_ai_config(settings_.sensitivity.toStdString(),
                                  settings_.absent_alert_seconds,
                                  settings_.seatmap_path.toStdString());

    connect(video_capture_.get(), &VideoCapture::frameReady, this,
            &SystemManager::on_frame_captured);

    // Gọi thẳng trên luồng camera (send_frame chỉ lọc nhịp + đặt frame vào ô chờ có khoá): đối tượng QThread
    // ZmqPublisher thuộc luồng UI, nối Queued thì mỗi frame 30 fps phải đi vòng qua hàng đợi sự kiện của giao diện
    connect(video_capture_.get(), &VideoCapture::frameReady,
            zmq_publisher_.get(), &ZmqPublisher::send_frame,
            Qt::DirectConnection);

    connect(video_capture_.get(), &VideoCapture::frameReady,
            image_processor_, &ImageProcessorWorker::process_frame,
            Qt::DirectConnection);

    if (!video_capture_->start_capture()) {
      stop_monitoring();
      return false;
    }
    zmq_publisher_->start_publish();

    is_monitoring_ = true;
    last_processed_frame_id_ = -1;
    frame_count_ = 0;
    last_frame_count_ = 0;
    current_fps_ = 0.0;
    last_fps_time_ = std::chrono::steady_clock::now();
    result_times_.clear();
    latency_ema_ms_ = 0.0;

    return true;

  } catch (const std::exception &e) {
    std::cerr << "Khong the khoi dong SystemManager: " << e.what() << std::endl;
    stop_monitoring();
    return false;
  }
}

void SystemManager::stop_monitoring() {
  // Giữ nguyên kết nối tới AI (subscriber + heartbeat) — chỉ dừng camera và gửi frame
  is_monitoring_ = false;

  if (video_capture_) {
    video_capture_->stop();
    video_capture_.reset();
  }
  if (zmq_publisher_) {
    zmq_publisher_->stop();
    zmq_publisher_.reset();
  }
}

void SystemManager::on_frame_captured(const cv::Mat &frame, int frame_id) {
  frame_w_ = frame.cols;
  frame_h_ = frame.rows;
  frame_count_++;
  capture_time_[frame_id % capture_time_.size()] = std::chrono::steady_clock::now();
}

void SystemManager::on_result_received(const exam::InferenceResult &result) {
  // Khi nhận được bất kỳ tin nhắn nào (kết quả thật hoặc heartbeat) đều tính là
  // online
  if (!is_backend_online_) {
    is_backend_online_ = true;
    emit backendStatusChanged(true);
  }

  // Reset timer
  heartbeat_timer_->start(5000);

  // Nếu chỉ là gói tin ping thì bỏ qua logic xử lý detection
  // Bỏ qua kết quả đến muộn sau khi đã bấm DỪNG LẠI
  if (result.is_heartbeat() || !is_monitoring_) {
    return;
  }

  if (result.frame_id() <= last_processed_frame_id_)
    return;
  last_processed_frame_id_ = result.frame_id();

  // Đo hiệu năng: độ trễ end-to-end (chụp -> nhận kết quả) và số nhịp AI/giây (cửa sổ 5s)
  auto now = std::chrono::steady_clock::now();
  if (frame_count_ > 0 &&
      static_cast<uint64_t>(result.frame_id()) + capture_time_.size() > frame_count_) {
    double latency = std::chrono::duration<double, std::milli>(
                         now - capture_time_[result.frame_id() % capture_time_.size()])
                         .count();
    latency_ema_ms_ = latency_ema_ms_ <= 0.0 ? latency : 0.8 * latency_ema_ms_ + 0.2 * latency;
  }
  result_times_.push_back(now);
  while (!result_times_.empty() && now - result_times_.front() > std::chrono::seconds(5))
    result_times_.pop_front();
  double ai_hz = 0.0;
  if (result_times_.size() >= 2) {
    double span = std::chrono::duration<double>(result_times_.back() - result_times_.front()).count();
    if (span > 0) ai_hz = (result_times_.size() - 1) / span;
  }
  emit aiMetricsUpdated(latency_ema_ms_, ai_hz);

  // Luôn chuyển tiếp (kể cả không có detection) để UI xóa khung cũ và cập nhật trạng thái AI
  emit processResultRequested(result);
}

void SystemManager::check_heartbeat() {
  // Nếu timer 3s bị kích hoạt tức là 3s qua không nhận được gì từ AI
  if (is_backend_online_) {
    is_backend_online_ = false;
    emit backendStatusChanged(false);
  }
}

double SystemManager::get_current_fps() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                     now - last_fps_time_)
                     .count();
  if (elapsed >= 1000) {
    current_fps_ = static_cast<double>(frame_count_ - last_frame_count_) *
                   1000.0 / elapsed;
    last_frame_count_ = frame_count_;
    last_fps_time_ = now;
  }
  return current_fps_;
}

uint64_t SystemManager::get_ai_count() const {
  return zmq_subscriber_ ? zmq_subscriber_->get_received_count() : 0;
}
