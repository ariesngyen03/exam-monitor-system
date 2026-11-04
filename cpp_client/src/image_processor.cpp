#include "image_processor.hpp"
#include "behavior_labels.hpp"
#include <QPainter>
#include <QDateTime>
#include <QStringList>
#include <algorithm>

namespace {
constexpr int kEmotionPerResult = 2;    // FER+ ~20 ms/mặt trên CPU; AI gửi ~2 nhịp/giây
constexpr float kFaceMargin = 0.15f;    // nới khung mặt mỗi bên cho giống ảnh huấn luyện FER+
}  // namespace

ImageProcessorWorker::ImageProcessorWorker(QObject* parent) : QObject(parent) {}

void ImageProcessorWorker::process_frame(const cv::Mat& frame, int frame_id) {
    if (frame.empty()) return;
    {
        QMutexLocker lock(&frame_mutex_);
        latest_frame_ = frame;
        latest_frame_id_ = frame_id;
    }

    // UI chưa vẽ xong frame trước (đang bận, VD nạp trang Lịch sử) -> bỏ frame này thay vì dồn hàng đợi gây trễ
    if (display_pending_.exchange(true)) return;

    // Thu nhỏ về đúng cỡ vùng vẽ NGAY TẠI luồng camera (~2 ms) để luồng UI chỉ chép ảnh 1:1, không phải co ảnh 1080p
    // mỗi lần vẽ. INTER_LINEAR: thu nhỏ < 2 lần nên đủ đẹp; INTER_AREA chậm gấp ~7 lần.
    const cv::Mat* src = &frame;
    const int dw = display_w_.load(), dh = display_h_.load();
    if (dw > 0 && dh > 0 && (frame.cols > dw || frame.rows > dh)) {
        const double s = std::min(static_cast<double>(dw) / frame.cols, static_cast<double>(dh) / frame.rows);
        cv::resize(frame, display_small_,
                   cv::Size(std::max(1, cvRound(frame.cols * s)), std::max(1, cvRound(frame.rows * s))), 0, 0,
                   cv::INTER_LINEAR);
        src = &display_small_;
    }
    // Chuyển thẳng sang Format_RGB32 (định dạng gốc của Qt trên Windows) trong 1 lượt
    QImage qimg(src->cols, src->rows, QImage::Format_RGB32);
    cv::Mat dst(src->rows, src->cols, CV_8UC4, qimg.bits(), static_cast<size_t>(qimg.bytesPerLine()));
    cv::cvtColor(*src, dst, cv::COLOR_BGR2BGRA);
    emit mainFrameProcessed(qimg, QSize(frame.cols, frame.rows));
}

void ImageProcessorWorker::annotate_emotions(std::vector<exam::Detection>& detections, const cv::Mat& frame_snap) {
    if (!enable_emotion_) {
        if (emotion_active_) emotion_.clear_all();   // vừa tắt: không giữ kết quả cũ
        emotion_active_ = false;
        return;
    }
    emotion_active_ = true;
    if (frame_snap.empty()) return;

    // Chỗ ngồi đã học, có người, AI chưa gửi cảm xúc (khi AI chạy --emotion thì giữ kết quả DeepFace)
    std::vector<exam::Detection*> present, faces;
    for (auto& d : detections) {
        if (d.desk_id() <= 0 || !d.emotion().empty()) continue;
        if (!d.is_present()) {
            emotion_.clear(d.track_id());            // rời chỗ: bỏ kết quả cũ
            continue;
        }
        present.push_back(&d);
        if (d.face_bbox_size() >= 4) faces.push_back(&d);
    }
    // Có người nhưng nhịp này không thấy mặt (cúi viết bài): vẫn hiện nhãn gần nhất trong kHoldSeconds
    auto apply_labels = [&]() {
        for (auto* d : present) {
            const std::string lab = emotion_.label(d->track_id());
            if (!lab.empty()) d->set_emotion(lab);
        }
    };
    if (faces.empty()) {
        apply_labels();
        return;
    }
    std::sort(faces.begin(), faces.end(),
              [](const exam::Detection* a, const exam::Detection* b) { return a->track_id() < b->track_id(); });

    // Khung mặt theo frame AI đã phân tích; frame_snap có thể mới hơn ~0.5 s — chấp nhận vì thí sinh ít di chuyển
    const cv::Rect frame_rect(0, 0, frame_snap.cols, frame_snap.rows);
    auto start = std::find_if(faces.begin(), faces.end(),
                              [this](const exam::Detection* d) { return d->track_id() >= emotion_rr_; });
    if (start == faces.end()) start = faces.begin();
    std::rotate(faces.begin(), start, faces.end());
    const int n = std::min<int>(kEmotionPerResult, static_cast<int>(faces.size()));
    for (int i = 0; i < n; ++i) {
        const auto* d = faces[i];
        const float w = d->face_bbox(2) - d->face_bbox(0), h = d->face_bbox(3) - d->face_bbox(1);
        cv::Rect r(cv::Point(static_cast<int>(d->face_bbox(0) - kFaceMargin * w), static_cast<int>(d->face_bbox(1) - kFaceMargin * h)),
                   cv::Point(static_cast<int>(d->face_bbox(2) + kFaceMargin * w), static_cast<int>(d->face_bbox(3) + kFaceMargin * h)));
        r &= frame_rect;
        if (r.area() > 0) emotion_.analyze(d->track_id(), frame_snap(r));
    }
    emotion_rr_ = faces[n - 1]->track_id() + 1;
    apply_labels();
}

QImage ImageProcessorWorker::make_snapshot(const exam::Detection& det, const cv::Mat& frame_snap) const {
    // Ưu tiên ảnh bằng chứng AI đã chụp (đúng khung hình lúc vi phạm, đã chú thích sẵn)
    if (!det.evidence_path().empty()) {
        QImage evidence(QString::fromStdString(det.evidence_path()));
        if (!evidence.isNull()) return evidence;
    }

    // Dự phòng: cắt vùng vi phạm từ khung hình mới nhất
    if (frame_snap.empty() || det.bbox_size() < 4) return {};
    QImage full_img(frame_snap.data, frame_snap.cols, frame_snap.rows,
                    static_cast<int>(frame_snap.step), QImage::Format_BGR888);

    const int margin = 20;
    QRect box(QPoint(static_cast<int>(det.bbox(0)), static_cast<int>(det.bbox(1))),
              QPoint(static_cast<int>(det.bbox(2)), static_cast<int>(det.bbox(3))));
    box = box.adjusted(-margin, -margin, margin, margin).intersected(full_img.rect());
    if (box.isEmpty()) return {};

    QImage cropped = full_img.copy(box);
    QPainter painter(&cropped);
    painter.setPen(QPen(Qt::red, 4));
    painter.drawRect(0, 0, cropped.width() - 1, cropped.height() - 1);
    painter.end();
    return cropped;
}

void ImageProcessorWorker::process_result(const exam::InferenceResult& result) {
    cv::Mat frame_snap;
    {
        QMutexLocker lock(&frame_mutex_);
        frame_snap = latest_frame_;
    }
    std::vector<exam::Detection> detections(result.detections().begin(), result.detections().end());
    annotate_emotions(detections, frame_snap);

    // Luôn gửi detections sang camera widget để vẽ overlay (kể cả rỗng -> xóa khung cũ)
    emit detectionsReady(detections);
    emit aiStatusReady(result.calibrating(), result.calib_progress(), result.n_people(), result.n_faces());

    for (auto& det : detections) {
        // AI tự chống trùng: chỉ bật new_alert khi mở sự cố mới hoặc chụp thêm bằng chứng (>= 3s/lần)
        if (!det.new_alert()) continue;

        QImage snapshot = make_snapshot(det, frame_snap);
        if (!snapshot.isNull()) {
            snapshot = snapshot.scaled(QSize(640, 360), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }

        QString position_str = behavior::seat_display(det.seat_label(), det.track_id());
        QString behavior_type = QString::fromStdString(det.behavior_type());
        // Điểm rủi ro là heuristic (không phải xác suất) -> chỉ hiển thị mức độ định tính
        QString level_str = behavior::risk_level_display(det.risk_level());

        QStringList violations;
        for (const auto& v : det.violations()) violations << QString::fromStdString(v).toHtmlEscaped();

        QString info = QString::fromUtf8("<b>Thời gian:</b> %1<br/><b>Vị trí:</b> %2<br/>"
                                         "<b>Nghi vấn:</b> <span style='color:#ef4444;'>%3</span><br/>"
                                         "<b>Mức độ:</b> %4")
                           .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                           .arg(position_str)
                           .arg(behavior::display_name(det.behavior_type()))
                           .arg(level_str);
        if (!violations.isEmpty()) {
            info += QString::fromUtf8("<br/><span style='color:#64748b;font-size:11px;'>%1</span>")
                        .arg(violations.join("<br/>"));
        }

        // Gửi qua tín hiệu để UI Thread xử lý QPixmap
        emit violationSnapshotProcessed(snapshot, info, behavior_type, position_str, level_str,
                                        QString::fromStdString(det.evidence_path()), det.new_incident());
    }
}
