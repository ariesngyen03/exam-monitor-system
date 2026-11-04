#pragma once

#include <QComboBox>
#include <QHash>
#include <QLabel>
#include <QWidget>

#include "ui_helpers.hpp"

// Trang "Thống kê": tổng hợp theo ngày từ báo cáo phiên + bộ đếm của phiên đang chạy
class StatsPage : public QWidget {
  Q_OBJECT
public:
  explicit StatsPage(QWidget *parent = nullptr);
  void set_data_dir(const QString &data_dir);

public slots:
  void reload();
  void add_live_alert(const QString &behavior_type);
  void reset_live();

private:
  void load_selected_date();
  void refresh_live();

  QString data_dir_;
  QComboBox *date_combo_ = nullptr;
  QLabel *val_sessions_ = nullptr;
  QLabel *val_incidents_ = nullptr;
  QLabel *val_hours_ = nullptr;
  QLabel *val_rate_ = nullptr;
  QLabel *val_precision_ = nullptr;
  ui::BarChart *chart_behavior_ = nullptr;
  ui::BarChart *chart_level_ = nullptr;
  ui::BarChart *chart_seat_ = nullptr;
  ui::BarChart *chart_live_ = nullptr;
  QLabel *live_caption_ = nullptr;
  QHash<QString, int> live_counts_;
};
