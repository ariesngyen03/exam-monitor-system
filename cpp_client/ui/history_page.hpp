#pragma once

#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStandardItemModel>
#include <QTableView>
#include <QWidget>

#include "../src/report_store.hpp"

// Trang "Lịch sử cảnh báo": đọc báo cáo phiên theo ngày (data/reports/*.json),
// cho phép giám thị xác minh sự cố (Human-in-the-loop) và xuất Biên bản PDF / CSV.
class HistoryPage : public QWidget {
  Q_OBJECT
public:
  explicit HistoryPage(QWidget *parent = nullptr);
  void set_data_dir(const QString &data_dir);

signals:
  void verificationUpdated();

public slots:
  void reload();

private:
  void load_selected_date();
  void populate_table(int keep_incident_idx = -1);
  void show_incident(int incident_idx);
  void apply_verification(const QString &status);
  int selected_incident_index() const;
  void export_csv();
  void export_pdf();

  QString data_dir_;
  reports::DayReport day_;

  QComboBox *date_combo_ = nullptr;
  QComboBox *filter_combo_ = nullptr;
  QLabel *summary_ = nullptr;
  QTableView *table_ = nullptr;
  QStandardItemModel *model_ = nullptr;
  QLabel *photo_ = nullptr;
  QLabel *info_ = nullptr;
  QLineEdit *note_edit_ = nullptr;
  QPushButton *btn_confirm_ = nullptr;
  QPushButton *btn_reject_ = nullptr;
  QPushButton *btn_open_image_ = nullptr;
  QPushButton *btn_open_folder_ = nullptr;
  QPushButton *btn_export_pdf_ = nullptr;
  QPushButton *btn_export_csv_ = nullptr;
};
