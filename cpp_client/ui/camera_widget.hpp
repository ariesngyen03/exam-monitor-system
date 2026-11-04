#pragma once

#include "exam_monitor.pb.h"
#include <QImage>
#include <QMutex>
#include <QPainter>
#include <QWidget>
#include <opencv2/opencv.hpp>

#include <vector>

/**
 * @brief Widget hiển thị camera với overlay kết quả AI.
 *
 * Nhận cv::Mat frame từ VideoCapture, convert sang QImage và vẽ lên widget.
 * Overlay bounding box, nhãn hành vi, track_id và desk_id.
 */
class CameraWidget : public QWidget {
  Q_OBJECT

public:
  explicit CameraWidget(QWidget *parent = nullptr);

signals:
  // Kích thước vùng vẽ (pixel thiết bị) đổi — luồng xử lý thu nhỏ frame theo cỡ này
  void displaySizeChanged(int width, int height);

public slots:
  // image có thể đã được thu nhỏ; source_size là kích thước frame gốc mà toạ độ box AI dựa theo
  void update_frame(const QImage &image, const QSize &source_size);
  void update_detections(const std::vector<exam::Detection> &detections);
  void update_ai_status(bool calibrating, float progress, int n_people, int n_faces);

protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private:
  void draw_calibration_banner(QPainter &painter, const QRect &area);
  void draw_detection(QPainter &painter, const exam::Detection &det, int x_off,
                      int y_off, double sx, double sy);

  QMutex mutex_;
  QImage current_frame_;
  QSize source_size_;
  std::vector<exam::Detection> detections_;
  bool calibrating_ = false;
  float calib_progress_ = 0.0f;
};
