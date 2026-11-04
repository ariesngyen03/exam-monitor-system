#include "camera_widget.hpp"
#include "behavior_labels.hpp"
#include <QPainterPath>
#include <QResizeEvent>
#include <algorithm>

CameraWidget::CameraWidget(QWidget *parent) : QWidget(parent) {
    setMinimumSize(640, 480);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setObjectName("CameraContainer");
}

void CameraWidget::update_frame(const QImage &image, const QSize &source_size) {
    QMutexLocker locker(&mutex_);
    current_frame_ = image;   // QImage chia sẻ ngầm, không cần copy
    source_size_ = source_size.isValid() ? source_size : image.size();
    update(); // Trigger repaint
}

void CameraWidget::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    const qreal dpr = devicePixelRatioF();
    emit displaySizeChanged(qRound(width() * dpr), qRound(height() * dpr));
}

void CameraWidget::update_detections(const std::vector<exam::Detection> &detections) {
    QMutexLocker locker(&mutex_);
    detections_ = detections;
    update();
}

void CameraWidget::update_ai_status(bool calibrating, float progress, int n_people, int n_faces) {
    Q_UNUSED(n_people);
    Q_UNUSED(n_faces);
    QMutexLocker locker(&mutex_);
    calibrating_ = calibrating;
    calib_progress_ = progress;
    update();
}

void CameraWidget::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    QPainter painter(this);

    QMutexLocker locker(&mutex_);

    if (current_frame_.isNull()) {
        // Hiển thị placeholder khi chưa có frame
        painter.fillRect(rect(), QColor("#f1f5f9"));
        painter.setPen(QColor("#94a3b8"));
        painter.setFont(QFont("Inter", 14, QFont::Medium));
        painter.drawText(rect(), Qt::AlignCenter,
                         QString::fromUtf8("Đang chờ kết nối camera..."));
        return;
    }

    // Ảnh đã được luồng camera thu nhỏ đúng cỡ vùng đích -> drawImage gần như chỉ là chép 1:1
    QSize target = source_size_.scaled(size(), Qt::KeepAspectRatio);
    int x_offset = (width() - target.width()) / 2;
    int y_offset = (height() - target.height()) / 2;
    QRect target_rect(QPoint(x_offset, y_offset), target);

    // Vẽ nền đen cho vùng thừa
    painter.fillRect(rect(), QColor("#0a0a0a"));
    painter.drawImage(target_rect, current_frame_);

    // Bật khử răng cưa chỉ cho phần vẽ khung overlay và chữ
    painter.setRenderHint(QPainter::Antialiasing, true);

    // Tính tỷ lệ scale cho overlay (box AI theo toạ độ frame gốc, không phải ảnh đã thu nhỏ)
    double scale_x =
        static_cast<double>(target.width()) / source_size_.width();
    double scale_y =
        static_cast<double>(target.height()) / source_size_.height();

    // Vẽ overlay kết quả AI
    for (const auto &det : detections_) {
        draw_detection(painter, det, x_offset, y_offset, scale_x, scale_y);
    }

    if (calibrating_) {
        draw_calibration_banner(painter, target_rect);
    }
}

void CameraWidget::draw_calibration_banner(QPainter &painter, const QRect &area) {
    const int bar_h = 44;
    QRect banner(area.left(), area.top(), area.width(), bar_h);
    painter.fillRect(banner, QColor(15, 23, 42, 190));

    QFont font("Inter", 10, QFont::Bold);
    font.insertSubstitution("Inter", "Segoe UI");
    painter.setFont(font);
    painter.setPen(Qt::white);
    QString text = QString::fromUtf8("ĐANG HỌC SƠ ĐỒ CHỖ NGỒI — thí sinh vui lòng ngồi yên  (%1%)")
                       .arg(static_cast<int>(calib_progress_ * 100.0f));
    painter.drawText(banner.adjusted(16, 0, -16, -8), Qt::AlignVCenter | Qt::AlignLeft, text);

    // Thanh tiến độ
    QRect track(banner.left(), banner.bottom() - 5, banner.width(), 6);
    painter.fillRect(track, QColor(255, 255, 255, 50));
    QRect fill = track;
    fill.setWidth(static_cast<int>(track.width() * std::clamp(calib_progress_, 0.0f, 1.0f)));
    painter.fillRect(fill, QColor("#3b82f6"));
}

void CameraWidget::draw_detection(QPainter &painter, const exam::Detection &det, int x_off,
                                  int y_off, double sx, double sy) {
    if (det.bbox_size() < 4)
        return;

    // Tọa độ bounding box đã scale
    int x1 = x_off + static_cast<int>(det.bbox(0) * sx);
    int y1 = y_off + static_cast<int>(det.bbox(1) * sy);
    int x2 = x_off + static_cast<int>(det.bbox(2) * sx);
    int y2 = y_off + static_cast<int>(det.bbox(3) * sy);
    int w = x2 - x1;
    int h = y2 - y1;

    const std::string &type = det.behavior_type();
    QColor color = behavior::color(type);

    // Chỗ trống / đang học: khung nét đứt, không nhãn hành vi
    QPen pen(color, (type == "Empty" || type == "Calibrating") ? 1 : 2);
    if (type == "Empty" || type == "Calibrating" || type == "Unassigned")
        pen.setStyle(Qt::DashLine);
    pen.setJoinStyle(Qt::MiterJoin);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(x1, y1, w, h);

    if (type == "Calibrating")
        return;

    // Nhãn: "D1-G2" cho chỗ bình thường, thêm hành vi + % khi có vi phạm
    QString label = QString::fromStdString(det.seat_label());
    if (type == "Unassigned") {
        label = behavior::display_name(type);
    } else if (behavior::is_violation(type)) {
        // Điểm rủi ro là heuristic -> chỉ hiện mức độ định tính, không hiện %
        label = QString("%1 · %2").arg(label, behavior::display_name(type).toUpper());
        if (!det.risk_level().empty())
            label += QString(" · %1").arg(behavior::risk_level_display(det.risk_level()).toUpper());
    }
    if (label.isEmpty())
        return;

    QFont font("Inter", 9, QFont::Bold);
    font.insertSubstitution("Inter", "Segoe UI");
    painter.setFont(font);
    QFontMetrics fm(font);
    int label_h = fm.height() + 6;
    int label_w = fm.horizontalAdvance(label) + 12;

    // Nền nhãn dính sát vào cạnh trên của bounding box (đẩy vào trong nếu tràn mép trên)
    int label_y = (y1 - label_h >= y_off) ? y1 - label_h : y1;
    QRect label_bg(x1, label_y, label_w, label_h);
    QColor bg = color;
    if (type == "Empty") bg.setAlpha(160);
    painter.fillRect(label_bg, bg);

    painter.setPen(Qt::white);
    painter.drawText(label_bg, Qt::AlignCenter, label);
}
