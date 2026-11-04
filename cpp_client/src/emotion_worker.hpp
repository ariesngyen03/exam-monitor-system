#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <array>
#include <chrono>
#include <string>
#include <unordered_map>

// Nhận diện cảm xúc FER+ (ONNX Model Zoo: emotion-ferplus-8) bằng OpenCV DNN — CHỈ THAM KHẢO.
// Không dùng để kết luận gian lận (thiếu cơ sở khoa học; dữ liệu nhạy cảm theo NĐ 13/2023). Bật/tắt ở trang Cài đặt.
// Mô hình nạp lười ở lần dùng đầu tiên, tìm theo find_model(). Không an toàn đa luồng: chỉ gọi từ 1 luồng.
class EmotionWorker {
public:
    static constexpr int kNumClasses = 8;

    // Phân tích 1 ảnh mặt (BGR, đã cắt) của chỗ ngồi track_id, cập nhật kết quả đã làm mượt.
    // Trả về false nếu bỏ qua (mô hình không nạp được, mặt quá nhỏ hoặc quá tối).
    bool analyze(int track_id, const cv::Mat& face_bgr);
    // Nhãn đã làm mượt (EMA xác suất); rỗng nếu chưa có, độ tin cậy thấp hoặc lần thấy mặt gần nhất đã quá
    // kHoldSeconds (thí sinh cúi viết bài nên mặt chỉ thấy lác đác — xoá ngay thì nhãn chỉ loé lên 0.5 s)
    std::string label(int track_id) const;
    void clear(int track_id) { ema_.erase(track_id); }
    void clear_all() { ema_.clear(); }

    // <thư mục exe>/models/fer_plus.onnx, sau đó <thư mục exe>/../python_ai/models/fer_plus.onnx
    static std::string find_model();

private:
    bool ensure_loaded();

    cv::dnn::Net net_;
    bool load_tried_ = false;
    struct State {
        std::array<float, kNumClasses> p{};
        std::chrono::steady_clock::time_point seen;
    };
    std::unordered_map<int, State> ema_;
    bool logged_first_ = false;
};
