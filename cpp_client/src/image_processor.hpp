#pragma once

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QString>
#include <opencv2/opencv.hpp>
#include "exam_monitor.pb.h"
#include <atomic>
#include <vector>
#include "emotion_worker.hpp"

class ImageProcessorWorker : public QObject {
    Q_OBJECT
public:
    explicit ImageProcessorWorker(QObject* parent = nullptr);
    // Gọi được từ luồng UI (cờ atomic); worker tự xoá kết quả cũ khi bị tắt
    void set_enable_emotion(bool enable) { enable_emotion_ = enable; }
    // Kích thước (pixel thiết bị) vùng vẽ video — frame được thu nhỏ về đúng cỡ này trước khi gửi sang luồng UI
    void set_display_size(int width, int height) { display_w_ = width; display_h_ = height; }
    // Luồng UI gọi sau khi nhận frame hiển thị; trước đó các frame mới bị bỏ qua (UI bận thì không dồn hàng đợi)
    void display_frame_consumed() { display_pending_ = false; }
    ~ImageProcessorWorker() override = default;

public slots:
    void process_frame(const cv::Mat& frame, int frame_id);
    void process_result(const exam::InferenceResult& result);

signals:
    // image đã thu nhỏ để hiển thị; source_size là kích thước frame gốc (toạ độ box của AI tính theo kích thước này)
    void mainFrameProcessed(const QImage& image, const QSize& source_size);
    // behavior_type là khóa gốc từ AI ("Leaning", "Turning Head"...) — UI tự dịch qua behavior_labels.hpp
    void violationSnapshotProcessed(const QImage& snapshot, const QString& info, const QString& behavior_type,
                                    const QString& position, const QString& risk_level,
                                    const QString& evidence_path, bool is_new_incident);
    void detectionsReady(const std::vector<exam::Detection>& detections);
    void aiStatusReady(bool calibrating, float progress, int n_people, int n_faces);

private:
    QImage make_snapshot(const exam::Detection& det, const cv::Mat& frame_snap) const;
    // Gắn nhãn cảm xúc FER+ (tham khảo) cho chỗ có mặt ở nhịp này, nếu AI chưa gửi sẵn
    void annotate_emotions(std::vector<exam::Detection>& detections, const cv::Mat& frame_snap);

    EmotionWorker emotion_;
    std::atomic<bool> enable_emotion_{true};    // mặc định BẬT (xem app_settings.hpp)
    bool emotion_active_ = false;
    int emotion_rr_ = 0;                        // xoay vòng track_id: mỗi nhịp chỉ phân tích vài mặt

    std::atomic<int> display_w_{0}, display_h_{0};
    std::atomic<bool> display_pending_{false};
    cv::Mat display_small_;                     // bộ đệm thu nhỏ, chỉ dùng trong luồng camera

    mutable QMutex frame_mutex_;
    cv::Mat latest_frame_;
    int latest_frame_id_ = -1;
};
