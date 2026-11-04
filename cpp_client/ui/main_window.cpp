#include "main_window.hpp"
#include <QDateTime>
#include <QDesktopServices>
#include <QFile>
#include <QFileInfo>
#include <QPainter>
#include <QPushButton>
#include <QStandardItemModel>
#include <QStyle>
#include <QUrl>

#include "../src/behavior_labels.hpp"
#include "history_page.hpp"
#include "live_page.hpp"
#include "settings_page.hpp"
#include "stats_page.hpp"
#include "ui_helpers.hpp"

namespace {
const char *kBorderNormal =
    "QFrame#CameraContainerFrame { border: 2px solid transparent; border-radius: 8px; }";
const char *kBorderAlert =
    "QFrame#CameraContainerFrame { border: 4px solid #ef4444; border-radius: 8px; }";
const int kLivePage = 1;
const int kHistoryPage = 2;
const int kStatsPage = 3;
} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  system_manager_ = new SystemManager(this);
  setup_ui();
  setup_connections();
  setup_timers();
}

MainWindow::~MainWindow() {
  if (system_manager_->is_monitoring()) {
    system_manager_->stop_monitoring();
  }
}

void MainWindow::setup_ui() {
  setWindowTitle(QString::fromUtf8("EXAM MONITOR SYSTEM - Dashboard"));
  setMinimumSize(1366, 768);
  resize(1440, 900);

  QFile styleFile(":/style.qss");
  if (styleFile.open(QFile::ReadOnly)) {
    setStyleSheet(QString::fromUtf8(styleFile.readAll()));
  }

  auto *central = new QWidget(this);
  setCentralWidget(central);
  auto *main_layout = new QHBoxLayout(central);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  // === SIDEBAR ===
  auto *sidebar = new QFrame(this);
  sidebar->setObjectName("Sidebar");
  sidebar->setFixedWidth(240);
  auto *sidebar_layout = new QVBoxLayout(sidebar);
  sidebar_layout->setContentsMargins(16, 24, 16, 24);
  sidebar_layout->setSpacing(8);

  // App branding (Logo) in Sidebar
  auto *brand_container = new QWidget();
  auto *brand_layout = new QHBoxLayout(brand_container);
  brand_layout->setContentsMargins(0, 0, 0, 24);
  auto *logo_lbl = new QLabel();
  logo_lbl->setPixmap(
      QPixmap(":/logo.jpg")
          .scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  logo_lbl->setFixedSize(64, 64);
  auto *brand_text = new QLabel("EXAM MONITOR\nSYSTEM");
  brand_text->setObjectName("BrandText");
  brand_layout->addWidget(logo_lbl);
  brand_layout->addWidget(brand_text);
  sidebar_layout->addWidget(brand_container);

  // Menu items
  page_titles_ = {QString::fromUtf8("Tổng quan"), QString::fromUtf8("Video trực tiếp"),
                  QString::fromUtf8("Lịch sử cảnh báo"), QString::fromUtf8("Thống kê"),
                  QString::fromUtf8("Cài đặt")};
  const QStringList icon_files = {":/home.png", ":/cam.png", ":/history.png",
                                  ":/monitorin.png", ":/setting.png"};

  for (int i = 0; i < page_titles_.size(); ++i) {
    auto *btn = new QPushButton(" " + page_titles_[i]);
    btn->setIcon(QIcon(icon_files[i]));
    btn->setIconSize(QSize(20, 20));
    btn->setObjectName("MenuButton");
    if (i == 0)
      btn->setProperty("active", true);
    sidebar_layout->addWidget(btn);
    sidebar_btns_.append(btn);
    connect(btn, &QPushButton::clicked, this, [this, i]() { switch_page(i); });
  }
  sidebar_layout->addStretch();

  // Status bottom
  auto *sys_status =
      new QLabel(QString::fromUtf8("Exam Monitor System\nPhiên bản 1.1.0"));
  sys_status->setStyleSheet("color: #64748b; font-size: 12px;");
  sidebar_layout->addWidget(sys_status);

  main_layout->addWidget(sidebar);

  // === RIGHT CONTENT ===
  auto *content_area = new QWidget(this);
  content_area->setObjectName("ContentArea");
  auto *content_layout = new QVBoxLayout(content_area);
  content_layout->setContentsMargins(0, 0, 0, 0);
  content_layout->setSpacing(0);

  // HEADER
  auto *header = new QFrame(this);
  header->setObjectName("Header");
  header->setFixedHeight(70);
  auto *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(24, 0, 24, 0);

  lbl_header_title_ = new QLabel(QString::fromUtf8(
      "Real-time Video Intelligence & Academic Integrity Surveillance"));
  lbl_header_title_->setStyleSheet(
      "color: #64748b; font-size: 14px; font-weight: 500;");
  header_layout->addWidget(lbl_header_title_);
  header_layout->addStretch();

  lbl_camera_status_ = new QLabel(QString::fromUtf8("⚪ Camera: Chưa bật"));
  lbl_camera_status_->setObjectName("StatusBadge");
  lbl_ai_status_ = new QLabel(QString::fromUtf8("🔴 AI: Chưa kết nối"));
  lbl_ai_status_->setObjectName("StatusBadge");
  lbl_ai_status_->setToolTip(QString::fromUtf8("Trạng thái AI server (python_ai/main.py) qua heartbeat mỗi giây"));
  lbl_time_ = new QLabel("00:00:00");
  lbl_time_->setObjectName("TimeBadge");

  btn_toggle_ = new QPushButton(QString::fromUtf8("BẮT ĐẦU"));
  btn_toggle_->setObjectName("BtnToggle");

  header_layout->addWidget(lbl_camera_status_, 0, Qt::AlignVCenter);
  header_layout->addWidget(lbl_ai_status_, 0, Qt::AlignVCenter);
  header_layout->addWidget(lbl_time_, 0, Qt::AlignVCenter);
  header_layout->addWidget(btn_toggle_, 0, Qt::AlignVCenter);

  content_layout->addWidget(header);

  // PAGES
  stacked_widget_ = new QStackedWidget(this);
  stacked_widget_->addWidget(build_dashboard()); // 0

  live_page_ = new LivePage();
  history_page_ = new HistoryPage();
  stats_page_ = new StatsPage();
  settings_page_ = new SettingsPage();
  stacked_widget_->addWidget(live_page_);        // 1
  stacked_widget_->addWidget(history_page_);     // 2
  stacked_widget_->addWidget(stats_page_);       // 3
  stacked_widget_->addWidget(settings_page_);    // 4

  const AppSettings &s = system_manager_->settings();
  settings_page_->load(s);
  history_page_->set_data_dir(s.data_dir);
  stats_page_->set_data_dir(s.data_dir);
  update_camera_title();

  content_layout->addWidget(stacked_widget_, 1);
  main_layout->addWidget(content_area, 1);

  connect(btn_toggle_, &QPushButton::clicked, this, &MainWindow::toggle_monitoring);
}

QWidget *MainWindow::build_dashboard() {
  auto *body_area = new QWidget(this);
  auto *body_layout = new QHBoxLayout(body_area);
  body_layout->setContentsMargins(24, 24, 24, 24);
  body_layout->setSpacing(24);

  // -- LEFT: VIDEO CARD --
  auto *video_card = ui::make_card(QString(), &dashboard_video_layout_);
  dashboard_video_layout_->setContentsMargins(16, 16, 16, 16);

  auto *cam_header = new QHBoxLayout();
  cam_title_ = new QLabel();
  cam_title_->setStyleSheet("font-weight: 800; font-size: 18px; color: #0f172a;");
  cam_badge_ = new QLabel(QString::fromUtf8("Ngoại tuyến"));
  cam_badge_->setObjectName("OfflineBadge");
  cam_header->addWidget(cam_title_);
  cam_header->addStretch();
  cam_header->addWidget(cam_badge_);
  dashboard_video_layout_->addLayout(cam_header);

  camera_container_ = new QFrame();
  camera_container_->setObjectName("CameraContainerFrame");
  auto *cam_grid = new QGridLayout(camera_container_);
  cam_grid->setContentsMargins(0, 0, 0, 0);
  camera_widget_ = new CameraWidget();
  cam_grid->addWidget(camera_widget_);
  dashboard_video_layout_->addWidget(camera_container_, 1);   // index 1 — xem switch_page()

  // Chỉ số hiệu năng thời gian thực
  lbl_metrics_ = new QLabel();
  lbl_metrics_->setStyleSheet("color: #64748b; font-size: 12px;");
  lbl_metrics_->setToolTip(QString::fromUtf8(
      "Camera: độ phân giải, định dạng (MJPG là đúng; YUY2 ở 1080p chỉ ~5-10 fps) và số frame/giây đọc từ camera\n"
      "AI: số kết quả phân tích nhận được mỗi giây\n"
      "Độ trễ: từ lúc chụp frame tới lúc nhận kết quả AI của frame đó"));
  dashboard_video_layout_->addWidget(lbl_metrics_);

  // Bottom AI info
  auto *ai_info_container = new QFrame();
  ai_info_container->setStyleSheet(
      "QFrame { background: #eff6ff; border-radius: 8px; border: 1px solid #bfdbfe; }");
  auto *ai_info_layout = new QHBoxLayout(ai_info_container);
  ai_info_layout->setContentsMargins(16, 12, 16, 12);
  ai_info_layout->setSpacing(12);

  auto *ai_icon = new QLabel();
  ai_icon->setPixmap(
      QPixmap(":/automation.png")
          .scaled(32, 32, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  ai_icon->setFixedSize(32, 32);
  ai_icon->setStyleSheet("border: none; background: transparent;");
  ai_info_layout->addWidget(ai_icon);

  ai_info_ = new QLabel();
  ai_info_->setStyleSheet("color: #1e40af; border: none; background: transparent;");
  ai_info_layout->addWidget(ai_info_);
  ai_info_layout->addStretch();
  dashboard_video_layout_->addWidget(ai_info_container);
  on_ai_status(false, 0.0f, 0, 0);

  body_layout->addWidget(video_card, 7);

  // -- RIGHT: SIDEBAR CARDS --
  auto *right_panel = new QWidget(this);
  right_panel->setFixedWidth(380);
  auto *right_layout = new QVBoxLayout(right_panel);
  right_layout->setContentsMargins(0, 0, 0, 0);
  right_layout->setSpacing(16);

  // Card 1: ALERTS
  QVBoxLayout *alert_layout = nullptr;
  auto *alert_card = ui::make_card(QString::fromUtf8("⚡ CẢNH BÁO MỚI NHẤT"), &alert_layout);
  auto *alert_content = new QHBoxLayout();
  detail_photo_ = new QLabel();
  detail_photo_->setFixedSize(140, 100);
  detail_photo_->setStyleSheet("background-color: #f1f5f9; border-radius: 6px; color: #94a3b8;");
  detail_photo_->setAlignment(Qt::AlignCenter);
  alert_content->addWidget(detail_photo_);

  detail_info_ = new QLabel(QString::fromUtf8("Chưa có cảnh báo..."));
  detail_info_->setWordWrap(true);
  detail_info_->setStyleSheet("color: #475569;");
  alert_content->addWidget(detail_info_, 1);
  alert_layout->addLayout(alert_content);

  auto *alert_btns = new QHBoxLayout();
  auto *btn_detail = new QPushButton(QString::fromUtf8("Mở ảnh bằng chứng"));
  btn_detail->setObjectName("BtnPrimary");
  auto *btn_handle = new QPushButton(QString::fromUtf8("Xóa danh sách"));
  btn_handle->setObjectName("BtnSecondary");
  alert_btns->addWidget(btn_detail);
  alert_btns->addWidget(btn_handle);
  alert_layout->addLayout(alert_btns);
  right_layout->addWidget(alert_card);

  // Card 2: STATS
  QVBoxLayout *stats_layout = nullptr;
  auto *stats_card = ui::make_card(QString::fromUtf8("📊 PHIÊN HIỆN TẠI"), &stats_layout);
  auto *stats_grid = new QGridLayout();
  const QStringList captions = {QString::fromUtf8("Sự cố"), QString::fromUtf8("Thí sinh"),
                                QString::fromUtf8("Thời gian")};
  for (int c = 0; c < captions.size(); ++c) {
    auto *cap = new QLabel(captions[c]);
    cap->setStyleSheet("color: #64748b; font-weight: 500;");
    cap->setAlignment(Qt::AlignCenter);
    stats_grid->addWidget(cap, 0, c);
  }
  lbl_total_alerts_ = new QLabel("0");
  lbl_total_alerts_->setObjectName("StatValueRed");
  lbl_people_ = new QLabel("—");
  lbl_people_->setObjectName("StatValueGreen");
  lbl_uptime_ = new QLabel("00:00:00");
  lbl_uptime_->setStyleSheet("color: #3b82f6; font-size: 24px; font-weight: bold;");
  for (auto *v : {lbl_total_alerts_, lbl_people_, lbl_uptime_}) v->setAlignment(Qt::AlignCenter);
  stats_grid->addWidget(lbl_total_alerts_, 1, 0);
  stats_grid->addWidget(lbl_people_, 1, 1);
  stats_grid->addWidget(lbl_uptime_, 1, 2);
  stats_layout->addLayout(stats_grid);
  right_layout->addWidget(stats_card);

  // Card 3: HISTORY
  QVBoxLayout *history_layout = nullptr;
  auto *history_card = ui::make_card(QString::fromUtf8("📜 CẢNH BÁO TRONG PHIÊN"), &history_layout);
  log_table_ = new QTableView();
  log_table_->setObjectName("LogTableLight");
  log_table_->verticalHeader()->setDefaultSectionSize(36);
  log_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
  log_table_->setSelectionMode(QAbstractItemView::SingleSelection);
  log_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  log_table_->setShowGrid(false);
  log_table_->verticalHeader()->setVisible(false);

  log_model_ = new QStandardItemModel(0, 4, this);
  log_model_->setHorizontalHeaderLabels(
      {QString::fromUtf8("Giờ"), QString::fromUtf8("Hành vi"),
       QString::fromUtf8("Vị trí"), QString::fromUtf8("Mức độ")});
  log_table_->setModel(log_model_);
  // Độ rộng cố định cho cột ngắn, cột Hành vi co giãn (ResizeToContents bị tính lúc bảng rỗng -> cắt chữ)
  auto *hh = log_table_->horizontalHeader();
  hh->setSectionResizeMode(QHeaderView::Fixed);
  hh->setSectionResizeMode(1, QHeaderView::Stretch);
  log_table_->setColumnWidth(0, 72);
  log_table_->setColumnWidth(2, 104);
  log_table_->setColumnWidth(3, 76);
  log_table_->setTextElideMode(Qt::ElideRight);
  history_layout->addWidget(log_table_);

  connect(log_table_->selectionModel(), &QItemSelectionModel::selectionChanged, this, [this]() {
    auto indices = log_table_->selectionModel()->selectedRows();
    if (indices.isEmpty()) return;
    // Ảnh + thông tin chi tiết được lưu kèm ô "Giờ" lúc thêm dòng
    auto *item = log_model_->item(indices.first().row(), 0);
    detail_info_->setText(item->data(Qt::UserRole).toString());
    detail_evidence_path_ = item->data(Qt::UserRole + 2).toString();
    QImage snap = item->data(Qt::UserRole + 1).value<QImage>();
    if (!snap.isNull()) {
      detail_photo_->setPixmap(QPixmap::fromImage(snap).scaled(
          detail_photo_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
      detail_photo_->setText(QString::fromUtf8("Không có ảnh"));
    }
  });

  right_layout->addWidget(history_card, 1);
  body_layout->addWidget(right_panel, 3);

  connect(btn_detail, &QPushButton::clicked, this, [this]() {
    if (detail_evidence_path_.isEmpty()) {
      QMessageBox::information(this, QString::fromUtf8("Bằng chứng"),
                               QString::fromUtf8("Chưa có ảnh bằng chứng cho cảnh báo này."));
      return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(detail_evidence_path_));
  });
  connect(btn_handle, &QPushButton::clicked, this, [this]() {
    log_model_->removeRows(0, log_model_->rowCount());
    detail_photo_->clear();
    detail_photo_->setText(QString::fromUtf8("Đã xóa"));
    detail_evidence_path_.clear();
    detail_info_->setText(QString::fromUtf8("Chưa có cảnh báo..."));
  });
  return body_area;
}

void MainWindow::setup_connections() {
  connect(system_manager_, &SystemManager::backendStatusChanged, this,
          &MainWindow::on_backend_status_changed);
  connect(system_manager_, &SystemManager::aiMetricsUpdated, this, &MainWindow::on_ai_metrics);

  auto *processor = system_manager_->get_image_processor();
  // Frame hiển thị đến từ luồng camera: nhận xong mới cho gửi frame kế tiếp (UI bận thì bỏ frame, không dồn hàng đợi)
  connect(processor, &ImageProcessorWorker::mainFrameProcessed, camera_widget_,
          [this, processor](const QImage &image, const QSize &source_size) {
            camera_widget_->update_frame(image, source_size);
            processor->display_frame_consumed();
          });
  connect(camera_widget_, &CameraWidget::displaySizeChanged, processor,
          &ImageProcessorWorker::set_display_size, Qt::DirectConnection);
  connect(processor, &ImageProcessorWorker::detectionsReady, camera_widget_,
          &CameraWidget::update_detections);
  connect(processor, &ImageProcessorWorker::detectionsReady, live_page_, &LivePage::update_seats);
  connect(processor, &ImageProcessorWorker::aiStatusReady, camera_widget_,
          &CameraWidget::update_ai_status);
  connect(processor, &ImageProcessorWorker::aiStatusReady, live_page_, &LivePage::update_status);
  connect(processor, &ImageProcessorWorker::aiStatusReady, this, &MainWindow::on_ai_status);
  connect(processor, &ImageProcessorWorker::violationSnapshotProcessed, this,
          &MainWindow::on_violation, Qt::QueuedConnection);

  connect(settings_page_, &SettingsPage::settingsSaved, this, [this](const AppSettings &s) {
    system_manager_->apply_settings(s);
    history_page_->set_data_dir(s.data_dir);
    stats_page_->set_data_dir(s.data_dir);
    update_camera_title();
  });
  connect(settings_page_, &SettingsPage::relearnRequested, this, [this]() {
    system_manager_->request_seat_relearn();
  });
  connect(history_page_, &HistoryPage::verificationUpdated, stats_page_, &StatsPage::reload);
}

void MainWindow::setup_timers() {
  auto *time_updater = new QTimer(this);
  connect(time_updater, &QTimer::timeout, this, &MainWindow::update_time);
  time_updater->start(1000);
  update_time();

  border_flash_timer_ = new QTimer(this);
  connect(border_flash_timer_, &QTimer::timeout, this,
          &MainWindow::toggle_border_color);
}

void MainWindow::switch_page(int index) {
  for (int j = 0; j < sidebar_btns_.size(); ++j) {
    sidebar_btns_[j]->setProperty("active", index == j);
    sidebar_btns_[j]->style()->unpolish(sidebar_btns_[j]);
    sidebar_btns_[j]->style()->polish(sidebar_btns_[j]);
  }

  // Chỉ có 1 widget camera: chuyển nó sang trang đang xem
  if (index == kLivePage) {
    live_page_->camera_slot()->addWidget(camera_container_, 1);
  } else if (camera_container_->parentWidget() == live_page_ ||
             live_page_->isAncestorOf(camera_container_)) {
    dashboard_video_layout_->insertWidget(1, camera_container_, 1);
  }

  if (index == kHistoryPage) history_page_->reload();
  if (index == kStatsPage) stats_page_->reload();

  stacked_widget_->setCurrentIndex(index);
  lbl_header_title_->setText(index == 0 ? QString::fromUtf8(
                                              "Real-time Video Intelligence & Academic Integrity Surveillance")
                                        : page_titles_[index]);
}

void MainWindow::update_camera_title() {
  const AppSettings &s = system_manager_->settings();
  QString source = s.video_label();
  if (QFileInfo(source).isFile()) source = QFileInfo(source).fileName();   // file video: chỉ hiện tên file
  if (source.size() > 40) source = "..." + source.right(37);
  cam_title_->setText(QString::fromUtf8("🔴 %1 · %2").arg(source.toUpper(), s.exam_room));
}

void MainWindow::set_camera_border(bool red) {
  camera_container_->setStyleSheet(red ? kBorderAlert : kBorderNormal);
  is_border_red_ = red;
}

void MainWindow::toggle_border_color() {
  if (flash_ticks_ > 0) {
    set_camera_border(!is_border_red_);
    flash_ticks_--;
  } else {
    border_flash_timer_->stop();
    set_camera_border(false);
  }
}

void MainWindow::on_violation(const QImage &snap, const QString &info, const QString &behavior_type,
                              const QString &position, const QString &risk_level,
                              const QString &evidence_path, bool is_new_incident) {
  if (!snap.isNull()) {
    detail_photo_->setPixmap(QPixmap::fromImage(snap).scaled(
        detail_photo_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
  }
  detail_info_->setText(info);
  detail_evidence_path_ = evidence_path;

  const std::string type = behavior_type.toStdString();
  auto *time_item = new QStandardItem(QDateTime::currentDateTime().toString("HH:mm:ss"));
  time_item->setData(info, Qt::UserRole);
  time_item->setData(snap, Qt::UserRole + 1);
  time_item->setData(evidence_path, Qt::UserRole + 2);

  QList<QStandardItem *> row_items;
  row_items << time_item << new QStandardItem(behavior::display_name(type))
            << new QStandardItem(position) << new QStandardItem(risk_level);
  for (auto *item : row_items) item->setForeground(QBrush(behavior::color(type)));
  if (!is_new_incident) {
    // Ảnh bổ sung cho sự cố đang diễn ra: chữ nghiêng để phân biệt với sự cố mới
    for (auto *item : row_items) {
      QFont f = item->font();
      f.setItalic(true);
      item->setFont(f);
    }
    row_items[1]->setText(row_items[1]->text() + QString::fromUtf8(" (tiếp diễn)"));
  }

  log_model_->insertRow(0, row_items);
  if (log_model_->rowCount() > 200) log_model_->removeRow(log_model_->rowCount() - 1);

  if (is_new_incident) {
    total_incidents_++;
    lbl_total_alerts_->setText(QString::number(total_incidents_));
    stats_page_->add_live_alert(behavior_type);
    flash_ticks_ = 8;
    if (!border_flash_timer_->isActive()) border_flash_timer_->start(250);
  }
}

void MainWindow::toggle_monitoring() {
  if (system_manager_->is_monitoring()) {
    system_manager_->stop_monitoring();
    border_flash_timer_->stop();
    set_camera_border(false);

    btn_toggle_->setText(QString::fromUtf8("BẮT ĐẦU"));
    btn_toggle_->setObjectName("BtnToggle");
    lbl_camera_status_->setText(QString::fromUtf8("⚪ Camera: Đã dừng"));
    cam_badge_->setText(QString::fromUtf8("Ngoại tuyến"));
    cam_badge_->setObjectName("OfflineBadge");
    settings_page_->set_monitoring(false);
    camera_widget_->update_detections({});
    camera_widget_->update_ai_status(false, 0.0f, 0, 0);
    on_ai_status(false, 0.0f, 0, 0);
    lbl_metrics_->clear();
    session_start_ = QDateTime();

    // AI đóng phiên sau ~5s không nhận frame rồi mới ghi báo cáo -> làm mới Lịch sử/Thống kê sau đó
    QTimer::singleShot(8000, this, [this]() {
      history_page_->reload();
      stats_page_->reload();
    });
  } else {
    if (!system_manager_->start_monitoring()) {
      QMessageBox::critical(
          this, QString::fromUtf8("Lỗi"),
          QString::fromUtf8("Không mở được nguồn video: %1.\nKiểm tra kết nối camera hoặc đổi camera / "
                            "nguồn video trong trang Cài đặt.")
              .arg(system_manager_->settings().video_label()));
      return;
    }
    // AI chưa sẵn sàng (đang nạp mô hình) -> chỉ báo ở khung thông tin, KHÔNG bật hộp thoại chặn giao diện;
    // khi AI lên, kết quả đầu tiên sẽ tự cập nhật lại dòng trạng thái
    on_ai_status(false, 0.0f, 0, 0);

    btn_toggle_->setText(QString::fromUtf8("DỪNG LẠI"));
    btn_toggle_->setObjectName("BtnToggleStop");
    lbl_camera_status_->setText(QString::fromUtf8("🟢 Camera: Đang chạy"));
    cam_badge_->setText(QString::fromUtf8("Trực tuyến"));
    cam_badge_->setObjectName("OnlineBadge");
    settings_page_->set_monitoring(true);

    // Phiên mới
    session_start_ = QDateTime::currentDateTime();
    total_incidents_ = 0;
    lbl_total_alerts_->setText("0");
    log_model_->removeRows(0, log_model_->rowCount());
    stats_page_->reset_live();
  }
  for (QWidget *w : {static_cast<QWidget *>(btn_toggle_), static_cast<QWidget *>(cam_badge_)}) {
    w->style()->unpolish(w);
    w->style()->polish(w);
  }
}

void MainWindow::on_ai_status(bool calibrating, float progress, int n_people,
                              int n_faces) {
  QString detail;
  if (!system_manager_->is_monitoring()) {
    detail = QString::fromUtf8("Bấm BẮT ĐẦU để giám sát. 15 giây đầu AI học sơ đồ chỗ ngồi (thí sinh ngồi yên).");
  } else if (!system_manager_->is_backend_online()) {
    detail = QString::fromUtf8("Camera đã bật — đang chờ AI khởi động (lần đầu có thể mất 20–60 giây)...");
  } else if (calibrating) {
    detail = QString::fromUtf8("Đang học sơ đồ chỗ ngồi... %1% — phát hiện %2 người")
                 .arg(static_cast<int>(progress * 100.0f))
                 .arg(n_people);
  } else {
    detail = QString::fromUtf8("Đang giám sát: %1 người · %2 khuôn mặt được phân tích")
                 .arg(n_people)
                 .arg(n_faces);
  }
  if (system_manager_->is_monitoring()) lbl_people_->setText(QString::number(n_people));
  ai_info_->setText(QString::fromUtf8("<b>AI hỗ trợ giám thị</b><br/><span "
                                      "style='color:#3b82f6;font-size:13px;'>%1</span>")
                        .arg(detail));
}

void MainWindow::on_ai_metrics(double latency_ms, double ai_hz) {
  ai_latency_ms_ = latency_ms;
  ai_hz_ = ai_hz;
}

void MainWindow::on_backend_status_changed(bool is_online) {
  is_backend_online_ = is_online;
  lbl_ai_status_->setText(is_online ? QString::fromUtf8("🟢 AI: Đã kết nối")
                                    : QString::fromUtf8("🔴 AI: Mất kết nối"));
}

void MainWindow::update_time() {
  lbl_time_->setText(QDateTime::currentDateTime().toString("HH:mm:ss"));

  if (session_start_.isValid()) {
    qint64 secs = session_start_.secsTo(QDateTime::currentDateTime());
    lbl_uptime_->setText(QString("%1:%2:%3")
                             .arg(secs / 3600, 2, 10, QChar('0'))
                             .arg((secs % 3600) / 60, 2, 10, QChar('0'))
                             .arg(secs % 60, 2, 10, QChar('0')));
  }

  if (system_manager_->is_monitoring()) {
    const QString format = system_manager_->get_frame_format();
    lbl_metrics_->setText(
        QString::fromUtf8("Camera %1×%2%3 · %4 fps  ·  AI %5 nhịp/giây  ·  Độ trễ AI %6")
            .arg(system_manager_->get_frame_width())
            .arg(system_manager_->get_frame_height())
            .arg(format.isEmpty() ? QString() : " " + format)
            .arg(system_manager_->get_current_fps(), 0, 'f', 1)
            .arg(ai_hz_, 0, 'f', 1)
            .arg(ai_latency_ms_ > 0 ? QString("%1 ms").arg(ai_latency_ms_, 0, 'f', 0) : QString("—")));
  }
}
