#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QWidget>

#include "../src/app_settings.hpp"

// Trang "Cài đặt": camera, phòng thi, giám thị, tốc độ gửi frame, thư mục dữ liệu; điều khiển AI
class SettingsPage : public QWidget {
  Q_OBJECT
public:
  explicit SettingsPage(QWidget *parent = nullptr);
  void load(const AppSettings &s);
  void set_monitoring(bool monitoring);

signals:
  void settingsSaved(const AppSettings &settings);
  void relearnRequested();

private:
  AppSettings collect() const;

  QSpinBox *camera_index_ = nullptr;
  QLineEdit *camera_source_ = nullptr;
  QLineEdit *room_ = nullptr;
  QLineEdit *proctor_ = nullptr;
  QSpinBox *send_fps_ = nullptr;
  QComboBox *camera_res_ = nullptr;
  QComboBox *sensitivity_ = nullptr;
  QSpinBox *absent_alert_ = nullptr;
  QCheckBox *enable_emotion_ = nullptr;
  QLineEdit *data_dir_ = nullptr;
  QLineEdit *seatmap_path_ = nullptr;
  QPushButton *btn_relearn_ = nullptr;
  QLabel *status_ = nullptr;
};
