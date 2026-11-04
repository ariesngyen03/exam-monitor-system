#include "settings_page.hpp"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QProcess>
#include <QTime>
#include <QUrl>
#include <QVBoxLayout>

#include "ui_helpers.hpp"

SettingsPage::SettingsPage(QWidget *parent) : QWidget(parent) {
  auto *root = new QVBoxLayout(this);
  root->setContentsMargins(24, 24, 24, 24);
  root->setSpacing(16);

  auto *row = new QHBoxLayout();
  row->setSpacing(16);

  // --- Phiên giám sát ---
  QVBoxLayout *l = nullptr;
  auto *card_session = ui::make_card(QString::fromUtf8("🎥 PHIÊN GIÁM SÁT"), &l);
  auto *form = new QFormLayout();
  form->setLabelAlignment(Qt::AlignRight);
  form->setSpacing(12);
  room_ = new QLineEdit();
  room_->setPlaceholderText("PHONG_THI_01");
  proctor_ = new QLineEdit();
  proctor_->setPlaceholderText(QString::fromUtf8("Để trống = tên tài khoản Windows"));
  camera_index_ = new QSpinBox();
  camera_index_->setRange(0, 10);
  camera_source_ = new QLineEdit();
  camera_source_->setPlaceholderText(QString::fromUtf8("Để trống = dùng camera theo chỉ số"));
  camera_source_->setToolTip(QString::fromUtf8(
      "Đường dẫn file video (demo), rtsp://... hoặc http://IP:8080/video (app IP Webcam)"));
  auto *btn_video = new QPushButton(QString::fromUtf8("File..."));
  btn_video->setObjectName("BtnSecondary");
  auto *source_row = new QHBoxLayout();
  source_row->addWidget(camera_source_, 1);
  source_row->addWidget(btn_video);
  connect(btn_video, &QPushButton::clicked, this, [this]() {
    QString f = QFileDialog::getOpenFileName(this, QString::fromUtf8("Chọn video"), QString(),
                                             "Video (*.mp4 *.avi *.mkv *.mov);;All (*.*)");
    if (!f.isEmpty()) camera_source_->setText(QDir::toNativeSeparators(f));
  });
  send_fps_ = new QSpinBox();
  send_fps_->setRange(1, 15);
  send_fps_->setSuffix(QString::fromUtf8(" frame/giây"));
  form->addRow(QString::fromUtf8("Phòng thi:"), room_);
  form->addRow(QString::fromUtf8("Giám thị:"), proctor_);
  form->addRow(QString::fromUtf8("Camera (chỉ số):"), camera_index_);
  form->addRow(QString::fromUtf8("Hoặc nguồn video:"), source_row);
  camera_res_ = new QComboBox();
  camera_res_->addItem(QString::fromUtf8("1080p (1920×1080) — khuyên dùng"), 1080);
  camera_res_->addItem(QString::fromUtf8("720p (1280×720) — máy yếu / camera cũ"), 720);
  camera_res_->setToolTip(QString::fromUtf8(
      "Độ phân giải yêu cầu camera. 1080p cho mặt người ở xa rõ hơn.\n"
      "Độ phân giải thật camera trả về hiện ở dòng thông số dưới khung video."));
  form->addRow(QString::fromUtf8("Độ phân giải:"), camera_res_);
  form->addRow(QString::fromUtf8("Gửi sang AI:"), send_fps_);
  enable_emotion_ = new QCheckBox(QString::fromUtf8("Hiện cảm xúc (chỉ tham khảo)"));
  enable_emotion_->setToolTip(QString::fromUtf8(
      "Nhận diện cảm xúc FER+ trên khuôn mặt nhìn thấy được. KHÔNG dùng để kết luận gian lận.\n"
      "Đây là dữ liệu cá nhân nhạy cảm (NĐ 13/2023) — chỉ bật khi thí sinh đã được thông báo."));
  form->addRow(QString::fromUtf8("Cảm xúc:"), enable_emotion_);
  l->addLayout(form);
  auto *hint = new QLabel(QString::fromUtf8(
      "• Phòng thi / giám thị được AI dùng khi BẮT ĐẦU phiên mới.<br/>"
      "• Đổi camera / nguồn video có hiệu lực ở lần BẮT ĐẦU tiếp theo.<br/>"
      "• AI phân tích 2 nhịp/giây; gửi 3–5 frame/giây là đủ, cao hơn chỉ tốn CPU."));
  hint->setWordWrap(true);
  hint->setStyleSheet("color: #64748b; font-size: 12px;");
  l->addWidget(hint);
  l->addStretch();
  row->addWidget(card_session, 1);

  // --- Dữ liệu & AI ---
  auto *card_data = ui::make_card(QString::fromUtf8("💾 DỮ LIỆU & CẤU HÌNH AI"), &l);
  auto *ai_form = new QFormLayout();
  ai_form->setLabelAlignment(Qt::AlignRight);
  ai_form->setSpacing(10);

  sensitivity_ = new QComboBox();
  sensitivity_->addItem(QString::fromUtf8("Thấp — Ít báo sai nhất (chỉ bắt lỗi rõ)"), "low");
  sensitivity_->addItem(QString::fromUtf8("Trung bình — Cân bằng (khuyên dùng)"), "medium");
  sensitivity_->addItem(QString::fromUtf8("Cao — Bắt nhạy nhất (phòng thi nghiêm ngặt)"), "high");
  sensitivity_->setToolTip(QString::fromUtf8(
      "Điều chỉnh ngưỡng điểm nghi vấn của mô hình SeatScorer + luật hành vi.\n"
      "Thay đổi có hiệu lực ngay lập tức trên AI Server mà không cần khởi động lại."));

  absent_alert_ = new QSpinBox();
  absent_alert_->setRange(5, 600);
  absent_alert_->setSingleStep(5);
  absent_alert_->setSuffix(QString::fromUtf8(" giây"));
  absent_alert_->setToolTip(QString::fromUtf8(
      "Số giây thí sinh rời khỏi ghế liên tục trước khi AI phát cảnh báo Vắng mặt / Rời vị trí."));

  ai_form->addRow(QString::fromUtf8("Độ nhạy AI:"), sensitivity_);
  ai_form->addRow(QString::fromUtf8("Báo rời ghế sau:"), absent_alert_);
  l->addLayout(ai_form);

  auto *seatmap_lbl = new QLabel(QString::fromUtf8(
      "<b>Sơ đồ ghế cố định (seatmap.json)</b><br/><span style='color:#64748b;font-size:12px;'>"
      "Để trống = AI tự học 15 giây đầu. Chọn file JSON cố định để vào giám sát ngay và đặt tên ghế S1, S2...</span>"));
  seatmap_lbl->setWordWrap(true);
  l->addWidget(seatmap_lbl);

  auto *seatmap_row = new QHBoxLayout();
  seatmap_path_ = new QLineEdit();
  seatmap_path_->setPlaceholderText(QString::fromUtf8("Để trống = AI tự động học vị trí ghế (15s đầu)"));
  auto *btn_seatmap_browse = new QPushButton(QString::fromUtf8("Chọn..."));
  btn_seatmap_browse->setObjectName("BtnSecondary");
  auto *btn_seatmap_clear = new QPushButton(QString::fromUtf8("Tự học"));
  btn_seatmap_clear->setObjectName("BtnSecondary");
  btn_seatmap_clear->setToolTip(QString::fromUtf8("Xoá đường dẫn sơ đồ cố định để AI tự học 15 giây đầu"));
  auto *btn_seatmap_edit = new QPushButton(QString::fromUtf8("Chỉnh sửa sơ đồ ghế..."));
  btn_seatmap_edit->setObjectName("BtnSecondary");
  seatmap_row->addWidget(seatmap_path_, 1);
  seatmap_row->addWidget(btn_seatmap_browse);
  seatmap_row->addWidget(btn_seatmap_clear);
  seatmap_row->addWidget(btn_seatmap_edit);
  l->addLayout(seatmap_row);
  l->addSpacing(8);

  auto *dir_row = new QHBoxLayout();
  data_dir_ = new QLineEdit();
  auto *btn_browse = new QPushButton(QString::fromUtf8("Chọn..."));
  btn_browse->setObjectName("BtnSecondary");
  auto *btn_open = new QPushButton(QString::fromUtf8("Mở"));
  btn_open->setObjectName("BtnSecondary");
  dir_row->addWidget(data_dir_, 1);
  dir_row->addWidget(btn_browse);
  dir_row->addWidget(btn_open);
  auto *dir_lbl = new QLabel(QString::fromUtf8(
      "Thư mục dữ liệu (evidence/, reports/, logs/ do AI ghi)<br/><span style='color:#64748b;font-size:12px;'>"
      "Phải trùng với --data-dir của AI server (mặc định: thư mục data/ của dự án).</span>"));
  dir_lbl->setWordWrap(true);
  dir_lbl->setStyleSheet("color: #334155; font-weight: 500;");
  l->addWidget(dir_lbl);
  l->addLayout(dir_row);
  l->addSpacing(8);

  auto *relearn_lbl = new QLabel(QString::fromUtf8(
      "<b>Học lại sơ đồ chỗ ngồi</b><br/><span style='color:#64748b;font-size:12px;'>"
      "Dùng khi camera bị xê dịch hoặc thí sinh đổi chỗ. AI xoá sơ đồ hiện tại và học lại trong 15 giây "
      "(thí sinh nên ngồi yên). Các sự cố đang mở sẽ được đóng.</span>"));
  relearn_lbl->setWordWrap(true);
  btn_relearn_ = new QPushButton(QString::fromUtf8("Học lại sơ đồ chỗ ngồi"));
  btn_relearn_->setObjectName("BtnPrimary");
  btn_relearn_->setEnabled(false);
  l->addWidget(relearn_lbl);
  l->addWidget(btn_relearn_, 0, Qt::AlignLeft);
  l->addStretch();
  row->addWidget(card_data, 1);

  root->addLayout(row);

  auto *actions = new QHBoxLayout();
  status_ = new QLabel();
  status_->setStyleSheet("color: #10b981; font-weight: 600;");
  auto *btn_save = new QPushButton(QString::fromUtf8("Lưu cài đặt"));
  btn_save->setObjectName("BtnToggle");
  actions->addWidget(status_, 1);
  actions->addWidget(btn_save);
  root->addLayout(actions);
  root->addStretch();

  connect(btn_save, &QPushButton::clicked, this, [this]() {
    emit settingsSaved(collect());
    status_->setText(QString::fromUtf8("✔ Đã lưu lúc %1").arg(QTime::currentTime().toString("HH:mm:ss")));
  });
  connect(btn_relearn_, &QPushButton::clicked, this, [this]() {
    emit relearnRequested();
    status_->setText(QString::fromUtf8("✔ Đã gửi yêu cầu học lại sơ đồ chỗ ngồi"));
  });
  connect(btn_seatmap_browse, &QPushButton::clicked, this, [this]() {
    QString f = QFileDialog::getOpenFileName(this, QString::fromUtf8("Chọn sơ đồ ghế JSON"),
                                             data_dir_->text(), "JSON (*.json);;All (*.*)");
    if (!f.isEmpty()) seatmap_path_->setText(QDir::toNativeSeparators(f));
  });
  connect(btn_seatmap_clear, &QPushButton::clicked, this, [this]() {
    seatmap_path_->clear();
    status_->setText(QString::fromUtf8("ℹ Đã chuyển sang chế độ AI tự học sơ đồ ghế (bấm Lưu cài đặt để áp dụng)"));
  });
  connect(btn_seatmap_edit, &QPushButton::clicked, this, [this]() {
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList scriptCandidates = {
        QDir(appDir).filePath("ai_server/seat_map_editor.py"),
        QDir(appDir).filePath("python_ai/seat_map_editor.py"),
        QDir(appDir).filePath("../python_ai/seat_map_editor.py"),
        QDir::current().filePath("python_ai/seat_map_editor.py"),
    };
    QString scriptPath;
    for (const QString &c : scriptCandidates) {
      if (QFileInfo::exists(c)) {
        scriptPath = QFileInfo(c).absoluteFilePath();
        break;
      }
    }
    if (scriptPath.isEmpty()) {
      status_->setText(QString::fromUtf8("⚠ Không tìm thấy công cụ python_ai/seat_map_editor.py"));
      return;
    }

    QStringList pyCandidates = {
        QDir(appDir).filePath("../python/python.exe"),
        QDir(appDir).filePath("python/python.exe"),
        QDir(appDir).filePath("ai_server/python/python.exe"),
        "C:/Users/hieu_/AppData/Local/Programs/Python/Python311/python.exe",
    };
    QString pyExe = "python";
    for (const QString &p : pyCandidates) {
      if (QFileInfo::exists(p)) {
        pyExe = QFileInfo(p).absoluteFilePath();
        break;
      }
    }

    QString outPath = QDir::fromNativeSeparators(seatmap_path_->text().trimmed());
    if (outPath.isEmpty()) {
      QString baseDir = data_dir_->text().trimmed().isEmpty() ? QDir::currentPath() : data_dir_->text().trimmed();
      QDir().mkpath(baseDir);
      outPath = QDir(baseDir).filePath("seatmap.json");
      seatmap_path_->setText(QDir::toNativeSeparators(outPath));
    }

    QString src = camera_source_->text().trimmed();
    if (src.isEmpty()) src = QString::number(camera_index_->value());

    QStringList args = {scriptPath, "--video", src, "--out", outPath};
    if (QFileInfo::exists(outPath)) {
      args << "--load" << outPath;
    }

    bool ok = QProcess::startDetached(pyExe, args, QFileInfo(scriptPath).absolutePath());
    if (ok) {
      status_->setText(QString::fromUtf8("✔ Đã mở trình chỉnh sửa sơ đồ ghế (bấm S trong cửa sổ để lưu)"));
    } else {
      status_->setText(QString::fromUtf8("⚠ Không khởi chạy được Python để mở trình chỉnh sửa sơ đồ ghế"));
    }
  });
  connect(btn_browse, &QPushButton::clicked, this, [this]() {
    QString dir = QFileDialog::getExistingDirectory(this, QString::fromUtf8("Chọn thư mục dữ liệu"),
                                                    data_dir_->text());
    if (!dir.isEmpty()) data_dir_->setText(QDir::toNativeSeparators(dir));
  });
  connect(btn_open, &QPushButton::clicked, this, [this]() {
    QDir().mkpath(data_dir_->text());
    QDesktopServices::openUrl(QUrl::fromLocalFile(data_dir_->text()));
  });
}

void SettingsPage::load(const AppSettings &s) {
  room_->setText(s.exam_room);
  proctor_->setText(s.proctor);
  camera_index_->setValue(s.camera_index);
  camera_source_->setText(QDir::toNativeSeparators(s.camera_source));
  send_fps_->setValue(s.send_fps);
  camera_res_->setCurrentIndex(s.camera_height >= 1080 ? 0 : 1);
  enable_emotion_->setChecked(s.enable_emotion);
  data_dir_->setText(QDir::toNativeSeparators(s.data_dir));
  int sensIdx = sensitivity_->findData(s.sensitivity.toLower());
  sensitivity_->setCurrentIndex(sensIdx >= 0 ? sensIdx : 1);
  absent_alert_->setValue(s.absent_alert_seconds);
  seatmap_path_->setText(s.seatmap_path.isEmpty() ? QString() : QDir::toNativeSeparators(s.seatmap_path));
}

void SettingsPage::set_monitoring(bool monitoring) { btn_relearn_->setEnabled(monitoring); }

AppSettings SettingsPage::collect() const {
  AppSettings s;
  s.exam_room = room_->text().trimmed().isEmpty() ? QString("PHONG_THI_01") : room_->text().trimmed();
  s.proctor = proctor_->text().trimmed();
  s.camera_index = camera_index_->value();
  s.camera_source = QDir::fromNativeSeparators(camera_source_->text().trimmed());
  s.send_fps = send_fps_->value();
  s.camera_height = camera_res_->currentData().toInt();
  s.enable_emotion = enable_emotion_->isChecked();
  s.data_dir = QDir::fromNativeSeparators(data_dir_->text().trimmed());
  s.sensitivity = sensitivity_->currentData().toString();
  s.absent_alert_seconds = absent_alert_->value();
  s.seatmap_path = QDir::fromNativeSeparators(seatmap_path_->text().trimmed());
  return s;
}
