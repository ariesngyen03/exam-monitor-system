#pragma once

#include <QLabel>
#include <QStandardItemModel>
#include <QTableView>
#include <QVBoxLayout>
#include <QWidget>
#include <vector>

#include "exam_monitor.pb.h"

// Trang "Video trực tiếp": khung camera lớn (widget camera được MainWindow chuyển qua lại giữa
// Tổng quan và trang này) + bảng trạng thái từng chỗ ngồi
class LivePage : public QWidget {
  Q_OBJECT
public:
  explicit LivePage(QWidget *parent = nullptr);
  QVBoxLayout *camera_slot() const { return camera_slot_; }

public slots:
  void update_seats(const std::vector<exam::Detection> &detections);
  void update_status(bool calibrating, float progress, int n_people, int n_faces);

private:
  QVBoxLayout *camera_slot_ = nullptr;
  QLabel *summary_ = nullptr;
  QTableView *table_ = nullptr;
  QStandardItemModel *model_ = nullptr;
};
