#pragma once

#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QTableView>
#include <QTimer>
#include <QVBoxLayout>
#include <QGraphicsDropShadowEffect>
#include <QPushButton>
#include <QStandardItemModel>
#include <QStackedWidget>
#include <QList>
#include <QDateTime>

#include "../src/system_manager.hpp"
#include "camera_widget.hpp"

class LivePage;
class HistoryPage;
class StatsPage;
class SettingsPage;

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private slots:
  void toggle_monitoring();
  void on_backend_status_changed(bool is_online);
  void update_time();
  void toggle_border_color();
  void on_ai_status(bool calibrating, float progress, int n_people, int n_faces);
  void on_ai_metrics(double latency_ms, double ai_hz);
  void on_violation(const QImage &snap, const QString &info, const QString &behavior_type,
                    const QString &position, const QString &risk_level, const QString &evidence_path,
                    bool is_new_incident);

private:
  void setup_ui();
  QWidget *build_dashboard();
  void setup_connections();
  void setup_timers();
  void switch_page(int index);
  void set_camera_border(bool red);
  void update_camera_title();

  // UI Components
  QStackedWidget *stacked_widget_ = nullptr;
  QList<QPushButton*> sidebar_btns_;
  QStringList page_titles_;
  QLabel *lbl_header_title_ = nullptr;

  QVBoxLayout *dashboard_video_layout_ = nullptr;
  QFrame *camera_container_ = nullptr;
  CameraWidget *camera_widget_ = nullptr;
  QLabel *cam_title_ = nullptr;
  QLabel *cam_badge_ = nullptr;
  QLabel *lbl_metrics_ = nullptr;
  QPushButton *btn_toggle_ = nullptr;

  QLabel *lbl_camera_status_ = nullptr;
  QLabel *lbl_ai_status_ = nullptr;
  QLabel *lbl_time_ = nullptr;
  QLabel *lbl_total_alerts_ = nullptr;
  QLabel *lbl_people_ = nullptr;
  QLabel *lbl_uptime_ = nullptr;
  int total_incidents_ = 0;

  // Detail Info
  QLabel *detail_photo_ = nullptr;
  QLabel *detail_info_ = nullptr;
  QString detail_evidence_path_;

  // AI status (dưới video)
  QLabel *ai_info_ = nullptr;

  // Log Table
  QTableView *log_table_ = nullptr;
  QStandardItemModel *log_model_ = nullptr;

  // Pages
  LivePage *live_page_ = nullptr;
  HistoryPage *history_page_ = nullptr;
  StatsPage *stats_page_ = nullptr;
  SettingsPage *settings_page_ = nullptr;

  // Core Logic
  SystemManager *system_manager_ = nullptr;

  // Timers & States
  QTimer *border_flash_timer_ = nullptr;
  bool is_backend_online_ = false;
  bool is_border_red_ = false;
  int flash_ticks_ = 0; // Đếm số lần nhấp nháy
  QDateTime session_start_;
  double ai_latency_ms_ = 0.0;
  double ai_hz_ = 0.0;
};
