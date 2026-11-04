#include "emotion_worker.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
// Thứ tự lớp đầu ra của emotion-ferplus-8
const char* const kLabels[EmotionWorker::kNumClasses] = {
    "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"};
constexpr int kInputSize = 64;         // đầu vào 1x1x64x64, ảnh xám, giá trị điểm ảnh 0..255
constexpr int kMinFacePx = 24;         // mặt nhỏ hơn -> ảnh phóng to quá vỡ, kết quả vô nghĩa
constexpr double kMinBrightness = 40;  // giống EMOTION_MIN_BRIGHTNESS phía Python: ảnh tối -> không đoán
constexpr float kAlpha = 0.5f;         // hệ số EMA cho kết quả mới
constexpr float kMinConfidence = 0.5f; // xác suất (đã làm mượt) tối thiểu để hiển thị nhãn
constexpr auto kHoldSeconds = std::chrono::seconds(10);   // giữ nhãn sau lần cuối thấy mặt
}  // namespace

std::string EmotionWorker::find_model() {
    QDir app(QCoreApplication::applicationDirPath());
    const QString candidates[] = {app.filePath("models/fer_plus.onnx"),
                                  app.filePath("../python_ai/models/fer_plus.onnx")};
    for (const auto& p : candidates)
        if (QFileInfo::exists(p)) return QDir::cleanPath(p).toStdString();
    return {};
}

bool EmotionWorker::ensure_loaded() {
    if (load_tried_) return !net_.empty();
    load_tried_ = true;
    const std::string path = find_model();
    if (path.empty()) {
        std::cerr << "[Emotion] Khong tim thay models/fer_plus.onnx - tat nhan dien cam xuc" << std::endl;
        return false;
    }
    try {
        net_ = cv::dnn::readNetFromONNX(path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "[Emotion] Loi nap " << path << ": " << e.what() << std::endl;
        net_ = cv::dnn::Net();
    }
    if (!net_.empty()) std::cerr << "[Emotion] Da nap " << path << std::endl;
    return !net_.empty();
}

bool EmotionWorker::analyze(int track_id, const cv::Mat& face_bgr) {
    if (face_bgr.empty() || std::min(face_bgr.cols, face_bgr.rows) < kMinFacePx) return false;
    if (!ensure_loaded()) return false;

    cv::Mat gray, resized;
    cv::cvtColor(face_bgr, gray, cv::COLOR_BGR2GRAY);
    if (cv::mean(gray)[0] < kMinBrightness) {
        ema_.erase(track_id);
        return false;
    }
    cv::resize(gray, resized, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_AREA);
    net_.setInput(cv::dnn::blobFromImage(resized));   // scale 1.0, không trừ mean: đúng tiền xử lý FER+

    cv::Mat out = net_.forward().reshape(1, 1);
    if (out.total() != static_cast<size_t>(kNumClasses)) return false;

    // Đầu ra là điểm chưa chuẩn hoá -> softmax
    std::array<float, kNumClasses> p{};
    const float* s = out.ptr<float>();
    const float mx = *std::max_element(s, s + kNumClasses);
    float sum = 0.f;
    for (int i = 0; i < kNumClasses; ++i) sum += (p[i] = std::exp(s[i] - mx));
    for (auto& v : p) v /= sum;

    auto it = ema_.find(track_id);
    if (it == ema_.end()) {
        it = ema_.emplace(track_id, State{p, {}}).first;
    } else {
        for (int i = 0; i < kNumClasses; ++i) it->second.p[i] = kAlpha * p[i] + (1.f - kAlpha) * it->second.p[i];
    }
    it->second.seen = std::chrono::steady_clock::now();
    if (!logged_first_) {
        logged_first_ = true;
        std::cerr << "[Emotion] Ket qua dau tien: cho #" << track_id << " -> " << label(track_id) << std::endl;
    }
    return true;
}

std::string EmotionWorker::label(int track_id) const {
    auto it = ema_.find(track_id);
    if (it == ema_.end() || std::chrono::steady_clock::now() - it->second.seen > kHoldSeconds) return {};
    const auto& p = it->second.p;
    const int best = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
    return p[best] >= kMinConfidence ? kLabels[best] : std::string();
}
