#include "history_page.hpp"

#include <QBuffer>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QPageLayout>
#include <QPageSize>
#include <QPdfWriter>
#include <QPixmap>
#include <QTextDocument>
#include <QTextStream>
#include <QUrl>
#include <QVBoxLayout>

#include "../src/behavior_labels.hpp"
#include "ui_helpers.hpp"

namespace {
QColor verify_color(const QString &status) {
  if (status == "confirmed") return QColor("#dc2626");
  if (status == "false_alarm") return QColor("#64748b");
  return QColor("#d97706");
}

QString csv_escape(const QString &s) {
  QString out = s;
  out.replace('"', "\"\"");
  return "\"" + out + "\"";
}
} // namespace

HistoryPage::HistoryPage(QWidget *parent) : QWidget(parent) {
  auto *root = new QVBoxLayout(this);
  root->setContentsMargins(24, 24, 24, 24);
  root->setSpacing(16);

  // Thanh chọn ngày + lọc trạng thái + xuất báo cáo
  auto *bar = new QHBoxLayout();
  auto *lbl = new QLabel(QString::fromUtf8("Ngày:"));
  lbl->setStyleSheet("font-weight: 600; color: #334155;");
  date_combo_ = new QComboBox();
  date_combo_->setMinimumWidth(140);

  auto *lbl_filter = new QLabel(QString::fromUtf8("Lọc:"));
  lbl_filter->setStyleSheet("font-weight: 600; color: #334155;");
  filter_combo_ = new QComboBox();
  filter_combo_->addItem(QString::fromUtf8("Tất cả sự cố"), "all");
  filter_combo_->addItem(QString::fromUtf8("⏳ Chờ duyệt"), "unverified");
  filter_combo_->addItem(QString::fromUtf8("✔ Vi phạm thật"), "confirmed");
  filter_combo_->addItem(QString::fromUtf8("✖ Báo sai"), "false_alarm");

  auto *btn_reload = new QPushButton(QString::fromUtf8("Làm mới"));
  btn_reload->setObjectName("BtnSecondary");
  summary_ = new QLabel();
  summary_->setStyleSheet("color: #64748b;");

  btn_export_pdf_ = new QPushButton(QString::fromUtf8("📄 Xuất Biên bản PDF"));
  btn_export_pdf_->setObjectName("BtnPrimary");
  btn_export_csv_ = new QPushButton(QString::fromUtf8("📊 Xuất CSV"));
  btn_export_csv_->setObjectName("BtnSecondary");

  bar->addWidget(lbl);
  bar->addWidget(date_combo_);
  bar->addSpacing(8);
  bar->addWidget(lbl_filter);
  bar->addWidget(filter_combo_);
  bar->addWidget(btn_reload);
  bar->addSpacing(12);
  bar->addWidget(summary_, 1);
  bar->addWidget(btn_export_csv_);
  bar->addWidget(btn_export_pdf_);
  root->addLayout(bar);

  auto *body = new QHBoxLayout();
  body->setSpacing(16);

  // Bảng sự cố
  QVBoxLayout *table_layout = nullptr;
  auto *table_card = ui::make_card(QString::fromUtf8("📜 SỰ CỐ ĐÃ GHI NHẬN"), &table_layout);
  table_ = new QTableView();
  table_->setObjectName("LogTableLight");
  table_->verticalHeader()->setVisible(false);
  table_->verticalHeader()->setDefaultSectionSize(36);
  table_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_->setShowGrid(false);
  model_ = new QStandardItemModel(0, 7, this);
  model_->setHorizontalHeaderLabels({QString::fromUtf8("Thời gian"), QString::fromUtf8("Phòng"),
                                     QString::fromUtf8("Vị trí"), QString::fromUtf8("Hành vi"),
                                     QString::fromUtf8("Mức độ"), QString::fromUtf8("Kéo dài"),
                                     QString::fromUtf8("Xác minh")});
  table_->setModel(model_);
  table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
  table_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
  table_layout->addWidget(table_);
  body->addWidget(table_card, 3);

  // Chi tiết + Xác minh của giám thị
  QVBoxLayout *detail_layout = nullptr;
  auto *detail_card = ui::make_card(QString::fromUtf8("🔍 CHI TIẾT & XÁC MINH GIÁM THỊ"), &detail_layout);
  detail_card->setMinimumWidth(400);
  photo_ = new QLabel(QString::fromUtf8("Chọn 1 sự cố để xem ảnh bằng chứng"));
  photo_->setAlignment(Qt::AlignCenter);
  photo_->setMinimumHeight(210);
  photo_->setStyleSheet("background-color: #f1f5f9; border-radius: 8px; color: #94a3b8;");
  info_ = new QLabel();
  info_->setWordWrap(true);
  info_->setTextFormat(Qt::RichText);
  info_->setStyleSheet("color: #475569;");

  auto *note_lbl = new QLabel(QString::fromUtf8("<b>Ghi chú giám thị / SBD thí sinh:</b>"));
  note_lbl->setStyleSheet("color: #334155; font-size: 12px;");
  note_edit_ = new QLineEdit();
  note_edit_->setPlaceholderText(QString::fromUtf8("Nhập SBD hoặc lý do (VD: Nhắc nhở lần 1, cúi nhặt bút...)"));
  note_edit_->setEnabled(false);

  auto *verify_btns = new QHBoxLayout();
  btn_confirm_ = new QPushButton(QString::fromUtf8("✔ Xác nhận vi phạm"));
  btn_confirm_->setObjectName("BtnToggle");
  btn_reject_ = new QPushButton(QString::fromUtf8("✖ Đánh dấu báo sai"));
  btn_reject_->setObjectName("BtnSecondary");
  btn_confirm_->setEnabled(false);
  btn_reject_->setEnabled(false);
  verify_btns->addWidget(btn_confirm_);
  verify_btns->addWidget(btn_reject_);

  auto *btns = new QHBoxLayout();
  btn_open_image_ = new QPushButton(QString::fromUtf8("Mở ảnh gốc"));
  btn_open_image_->setObjectName("BtnPrimary");
  btn_open_folder_ = new QPushButton(QString::fromUtf8("Mở thư mục"));
  btn_open_folder_->setObjectName("BtnSecondary");
  btn_open_image_->setEnabled(false);
  btn_open_folder_->setEnabled(false);
  btns->addWidget(btn_open_image_);
  btns->addWidget(btn_open_folder_);

  detail_layout->addWidget(photo_);
  detail_layout->addWidget(info_);
  detail_layout->addSpacing(4);
  detail_layout->addWidget(note_lbl);
  detail_layout->addWidget(note_edit_);
  detail_layout->addLayout(verify_btns);
  detail_layout->addStretch();
  detail_layout->addLayout(btns);
  body->addWidget(detail_card, 2);

  root->addLayout(body, 1);

  connect(btn_reload, &QPushButton::clicked, this, &HistoryPage::reload);
  connect(date_combo_, &QComboBox::currentIndexChanged, this, [this](int) { load_selected_date(); });
  connect(filter_combo_, &QComboBox::currentIndexChanged, this, [this](int) { populate_table(); });
  connect(table_->selectionModel(), &QItemSelectionModel::selectionChanged, this, [this]() {
    int idx = selected_incident_index();
    if (idx >= 0) show_incident(idx);
  });
  connect(btn_confirm_, &QPushButton::clicked, this, [this]() { apply_verification("confirmed"); });
  connect(btn_reject_, &QPushButton::clicked, this, [this]() { apply_verification("false_alarm"); });
  connect(note_edit_, &QLineEdit::returnPressed, this, [this]() {
    int idx = selected_incident_index();
    if (idx >= 0) apply_verification(day_.incidents[idx].verify_status);
  });
  connect(btn_open_image_, &QPushButton::clicked, this, [this]() {
    int idx = selected_incident_index();
    if (idx < 0) return;
    QDesktopServices::openUrl(QUrl::fromLocalFile(day_.incidents[idx].screenshot_path));
  });
  connect(btn_open_folder_, &QPushButton::clicked, this, [this]() {
    int idx = selected_incident_index();
    if (idx < 0) return;
    QFileInfo fi(day_.incidents[idx].screenshot_path);
    QDesktopServices::openUrl(QUrl::fromLocalFile(fi.absolutePath()));
  });
  connect(btn_export_csv_, &QPushButton::clicked, this, &HistoryPage::export_csv);
  connect(btn_export_pdf_, &QPushButton::clicked, this, &HistoryPage::export_pdf);
}

int HistoryPage::selected_incident_index() const {
  auto rows = table_->selectionModel()->selectedRows();
  if (rows.isEmpty()) return -1;
  auto *item = model_->item(rows.first().row(), 0);
  if (!item) return -1;
  int idx = item->data(Qt::UserRole).toInt();
  return (idx >= 0 && idx < day_.incidents.size()) ? idx : -1;
}

void HistoryPage::set_data_dir(const QString &data_dir) {
  data_dir_ = data_dir;
  reload();
}

void HistoryPage::reload() {
  const QDate keep = date_combo_->currentData().toDate();
  date_combo_->blockSignals(true);
  date_combo_->clear();
  for (const QDate &d : reports::available_dates(data_dir_))
    date_combo_->addItem(d.toString("dd/MM/yyyy"), d);
  int idx = keep.isValid() ? date_combo_->findData(keep) : -1;
  date_combo_->setCurrentIndex(idx >= 0 ? idx : 0);
  date_combo_->blockSignals(false);
  load_selected_date();
}

void HistoryPage::load_selected_date() {
  const QDate date = date_combo_->currentData().toDate();
  if (!date.isValid()) {
    day_ = {};
    model_->removeRows(0, model_->rowCount());
    photo_->clear();
    photo_->setText(QString::fromUtf8("Chọn 1 sự cố để xem ảnh bằng chứng"));
    info_->clear();
    note_edit_->clear();
    note_edit_->setEnabled(false);
    btn_confirm_->setEnabled(false);
    btn_reject_->setEnabled(false);
    btn_open_image_->setEnabled(false);
    btn_open_folder_->setEnabled(false);
    summary_->setText(QString::fromUtf8("Chưa có báo cáo nào trong %1 — báo cáo được ghi khi kết thúc phiên giám sát.")
                          .arg(reports::reports_dir(data_dir_)));
    return;
  }
  day_ = reports::load_day(data_dir_, date);
  if (!day_.error.isEmpty()) {
    summary_->setText(day_.error);
    model_->removeRows(0, model_->rowCount());
    return;
  }
  populate_table();
}

void HistoryPage::populate_table(int keep_incident_idx) {
  model_->removeRows(0, model_->rowCount());
  photo_->clear();
  photo_->setText(QString::fromUtf8("Chọn 1 sự cố để xem ảnh bằng chứng"));
  info_->clear();
  note_edit_->clear();
  note_edit_->setEnabled(false);
  btn_confirm_->setEnabled(false);
  btn_reject_->setEnabled(false);
  btn_open_image_->setEnabled(false);
  btn_open_folder_->setEnabled(false);

  int n_confirmed = 0, n_false = 0, n_unverified = 0;
  for (const auto &inc : day_.incidents) {
    if (inc.verify_status == "confirmed") n_confirmed++;
    else if (inc.verify_status == "false_alarm") n_false++;
    else n_unverified++;
  }
  summary_->setText(
      QString::fromUtf8("%1 phiên · %2 sự cố (✔ %3 vi phạm · ✖ %4 báo sai · ⏳ %5 chờ duyệt)")
          .arg(day_.sessions.size())
          .arg(day_.incidents.size())
          .arg(n_confirmed)
          .arg(n_false)
          .arg(n_unverified));

  const QString filter = filter_combo_->currentData().toString();
  int select_table_row = -1;

  for (int i = 0; i < day_.incidents.size(); ++i) {
    const auto &inc = day_.incidents[i];
    if (filter != "all" && inc.verify_status != filter) continue;

    const std::string type = inc.behavior_type.toStdString();
    auto *time_item = new QStandardItem(inc.detected_at.mid(11));
    time_item->setData(i, Qt::UserRole);

    auto *verify_item = new QStandardItem(reports::verify_status_display(inc.verify_status));
    verify_item->setForeground(verify_color(inc.verify_status));

    QList<QStandardItem *> row;
    row << time_item
        << new QStandardItem(inc.room)
        << new QStandardItem(behavior::seat_display(inc.seat_label.toStdString(), 0))
        << new QStandardItem(behavior::display_name(type))
        << new QStandardItem(behavior::risk_level_display(inc.level.toStdString()))
        << new QStandardItem(QString("%1 s").arg(inc.duration_seconds, 0, 'f', 1))
        << verify_item;
    row[3]->setForeground(behavior::color(type));
    if (i == keep_incident_idx) select_table_row = model_->rowCount();
    model_->appendRow(row);
  }

  if (select_table_row >= 0) {
    table_->selectRow(select_table_row);
  } else if (model_->rowCount() > 0) {
    table_->selectRow(0);
  }
}

void HistoryPage::show_incident(int incident_idx) {
  if (incident_idx < 0 || incident_idx >= day_.incidents.size()) return;
  const auto &inc = day_.incidents[incident_idx];

  QPixmap pix(inc.screenshot_path);
  if (!pix.isNull()) {
    photo_->setPixmap(pix.scaled(photo_->width(), photo_->height(), Qt::KeepAspectRatio,
                                 Qt::SmoothTransformation));
  } else {
    photo_->setText(QString::fromUtf8("Không tìm thấy ảnh bằng chứng"));
  }
  QStringList viols;
  for (const auto &v : inc.violations) viols << v.toHtmlEscaped();
  info_->setText(
      QString::fromUtf8("<b>Thời điểm:</b> %1<br/><b>Phòng:</b> %2 · <b>Giám thị:</b> %3<br/>"
                        "<b>Vị trí:</b> %4<br/>"
                        "<b>Hành vi:</b> <span style='color:%5;'>%6</span><br/>"
                        "<b>Mức độ:</b> %7 — %8<br/><b>Kéo dài:</b> %9 s<br/>"
                        "<b>Trạng thái:</b> <span style='color:%10;font-weight:600;'>%11</span><br/><br/>"
                        "<span style='color:#64748b;font-size:11px;'>%12</span>")
          .arg(inc.detected_at, inc.room, inc.proctor.isEmpty() ? QString("—") : inc.proctor,
               behavior::seat_display(inc.seat_label.toStdString(), 0),
               behavior::color(inc.behavior_type.toStdString()).name(),
               behavior::display_name(inc.behavior_type.toStdString()),
               behavior::risk_level_display(inc.level.toStdString()), inc.assessment.toHtmlEscaped(),
               QString::number(inc.duration_seconds, 'f', 1),
               verify_color(inc.verify_status).name(),
               reports::verify_status_display(inc.verify_status),
               viols.join("<br/>")));

  note_edit_->setEnabled(true);
  note_edit_->setText(inc.proctor_note);
  btn_confirm_->setEnabled(true);
  btn_reject_->setEnabled(true);

  const bool has_file = QFileInfo::exists(inc.screenshot_path);
  btn_open_image_->setEnabled(has_file);
  btn_open_folder_->setEnabled(has_file);
}

void HistoryPage::apply_verification(const QString &status) {
  int idx = selected_incident_index();
  if (idx < 0) return;

  auto &inc = day_.incidents[idx];
  const QString note = note_edit_->text().trimmed();
  if (reports::update_incident_verification(data_dir_, day_.date, inc.session_id,
                                            inc.detected_at, inc.seat_label, status, note)) {
    inc.verify_status = status;
    inc.proctor_note = note;
    populate_table(idx);
    emit verificationUpdated();
  } else {
    QMessageBox::warning(this, QString::fromUtf8("Lỗi lưu xác minh"),
                         QString::fromUtf8("Không thể cập nhật file báo cáo JSON của ngày %1.")
                             .arg(day_.date.toString("dd/MM/yyyy")));
  }
}

void HistoryPage::export_csv() {
  if (!day_.date.isValid() || day_.incidents.isEmpty()) {
    QMessageBox::information(this, QString::fromUtf8("Xuất CSV"),
                             QString::fromUtf8("Không có dữ liệu sự cố nào trong ngày đang chọn để xuất."));
    return;
  }

  const QString defaultName = QDir(data_dir_).filePath(
      QString("BaoCao_GiamSat_%1.csv").arg(day_.date.toString("yyyy-MM-dd")));
  const QString path = QFileDialog::getSaveFileName(
      this, QString::fromUtf8("Xuất danh sách sự cố ra CSV"), defaultName, "CSV (*.csv)");
  if (path.isEmpty()) return;

  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
    QMessageBox::critical(this, QString::fromUtf8("Lỗi"),
                          QString::fromUtf8("Không thể ghi file: %1").arg(path));
    return;
  }

  // Ghi UTF-8 BOM để Microsoft Excel hiển thị đúng tiếng Việt
  f.write("\xEF\xBB\xBF");
  QTextStream out(&f);
  out.setEncoding(QStringConverter::Utf8);

  out << "STT,Phien,Phong thi,Giam thi,Thoi diem phat hien,Vi tri ghe,Loai hanh vi,Muc do,"
         "Thoi luong (giay),Trang thai xac minh,Ghi chu giam thi,Chi tiet vi pham,Anh bang chung\n";

  const QString filter = filter_combo_->currentData().toString();
  int stt = 0;
  for (const auto &inc : day_.incidents) {
    if (filter != "all" && inc.verify_status != filter) continue;
    stt++;
    out << stt << ","
        << csv_escape(inc.session_id) << ","
        << csv_escape(inc.room) << ","
        << csv_escape(inc.proctor) << ","
        << csv_escape(inc.detected_at) << ","
        << csv_escape(behavior::seat_display(inc.seat_label.toStdString(), 0)) << ","
        << csv_escape(behavior::display_name(inc.behavior_type.toStdString())) << ","
        << csv_escape(behavior::risk_level_display(inc.level.toStdString())) << ","
        << QString::number(inc.duration_seconds, 'f', 1) << ","
        << csv_escape(reports::verify_status_display(inc.verify_status)) << ","
        << csv_escape(inc.proctor_note) << ","
        << csv_escape(inc.violations.join("; ")) << ","
        << csv_escape(inc.screenshot_path) << "\n";
  }
  f.close();
  QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void HistoryPage::export_pdf() {
  if (!day_.date.isValid()) {
    QMessageBox::information(this, QString::fromUtf8("Xuất Biên bản PDF"),
                             QString::fromUtf8("Chưa có báo cáo ngày để xuất PDF."));
    return;
  }

  const QString defaultName = QDir(data_dir_).filePath(
      QString("BienBan_GiamSat_%1.pdf").arg(day_.date.toString("yyyy-MM-dd")));
  const QString pdfPath = QFileDialog::getSaveFileName(
      this, QString::fromUtf8("Xuất Biên bản Giám sát PDF"), defaultName, "PDF (*.pdf)");
  if (pdfPath.isEmpty()) return;

  QStringList rooms, proctors;
  double total_sec = 0.0;
  for (const auto &s : day_.sessions) {
    if (!s.room.isEmpty() && !rooms.contains(s.room)) rooms << s.room;
    if (!s.proctor.isEmpty() && !proctors.contains(s.proctor)) proctors << s.proctor;
    total_sec += s.duration_seconds;
  }

  int n_confirmed = 0, n_false = 0, n_unverified = 0;
  for (const auto &inc : day_.incidents) {
    if (inc.verify_status == "confirmed") n_confirmed++;
    else if (inc.verify_status == "false_alarm") n_false++;
    else n_unverified++;
  }
  const int n_verified = n_confirmed + n_false;
  const QString precision_str = n_verified > 0
                                    ? QString("%1% (%2/%3)")
                                          .arg(100.0 * n_confirmed / n_verified, 0, 'f', 1)
                                          .arg(n_confirmed)
                                          .arg(n_verified)
                                    : QString::fromUtf8("Chưa chấm duyệt");

  const QString filter = filter_combo_->currentData().toString();
  QString rows_html;
  int stt = 0;
  for (const auto &inc : day_.incidents) {
    if (filter != "all" && inc.verify_status != filter) continue;
    // Mặc định trong biên bản PDF: nếu chọn "all", vẫn xuất tất cả kèm trạng thái rõ ràng
    stt++;

    QString img_html = QString::fromUtf8("<i>Không có ảnh</i>");
    if (QFileInfo::exists(inc.screenshot_path)) {
      QPixmap pix(inc.screenshot_path);
      if (!pix.isNull()) {
        QPixmap scaled = pix.scaled(160, 110, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        QByteArray ba;
        QBuffer buf(&ba);
        buf.open(QIODevice::WriteOnly);
        scaled.save(&buf, "JPG", 82);
        img_html = QString("<img width='150' src='data:image/jpeg;base64,%1'/>")
                       .arg(QString::fromLatin1(ba.toBase64()));
      }
    }

    rows_html += QString::fromUtf8(
                     "<tr>"
                     "<td align='center'>%1</td>"
                     "<td><b>%2</b><br/><span style='color:#64748b;'>Phòng: %3</span></td>"
                     "<td align='center'><b>%4</b></td>"
                     "<td><b>%5</b><br/><span style='color:#475569;font-size:8pt;'>%6</span></td>"
                     "<td align='center'>%7<br/>(%8s)</td>"
                     "<td><b>%9</b><br/><i>%10</i></td>"
                     "<td align='center'>%11</td>"
                     "</tr>")
                     .arg(stt)
                     .arg(inc.detected_at.toHtmlEscaped(), inc.room.toHtmlEscaped())
                     .arg(behavior::seat_display(inc.seat_label.toStdString(), 0).toHtmlEscaped())
                     .arg(behavior::display_name(inc.behavior_type.toStdString()).toHtmlEscaped(),
                          inc.violations.join("; ").toHtmlEscaped())
                     .arg(behavior::risk_level_display(inc.level.toStdString()).toHtmlEscaped(),
                          QString::number(inc.duration_seconds, 'f', 1))
                     .arg(reports::verify_status_display(inc.verify_status).toHtmlEscaped(),
                          inc.proctor_note.isEmpty() ? QString("—") : inc.proctor_note.toHtmlEscaped())
                     .arg(img_html);
  }

  if (rows_html.isEmpty()) {
    rows_html = QString::fromUtf8(
        "<tr><td colspan='7' align='center'><i>Không có sự cố nào khớp bộ lọc hiện tại.</i></td></tr>");
  }

  const QString html =
      QString::fromUtf8(
          "<html><head><meta charset='utf-8'/>"
          "<style>"
          "body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.5pt; color: #0f172a; }"
          "h1 { font-size: 15pt; margin: 0 0 4px 0; text-align: center; color: #1e3a8a; }"
          "h2 { font-size: 11pt; margin: 0 0 14px 0; text-align: center; color: #475569; font-weight: normal; }"
          "table.meta { width: 100%; border-collapse: collapse; margin-bottom: 12px; }"
          "table.meta td { padding: 4px 8px; font-size: 9.5pt; }"
          "table.inc { width: 100%; border-collapse: collapse; margin-top: 8px; }"
          "table.inc th { background-color: #e2e8f0; color: #0f172a; padding: 6px; border: 1px solid #cbd5e1; font-size: 9pt; }"
          "table.inc td { padding: 6px; border: 1px solid #cbd5e1; vertical-align: middle; font-size: 8.5pt; }"
          "table.sig { width: 100%; margin-top: 28px; border-collapse: collapse; }"
          "table.sig td { text-align: center; vertical-align: top; width: 50%; }"
          "</style></head><body>"
          "<h1>BIÊN BẢN TỔNG HỢP GIÁM SÁT PHÒNG THI</h1>"
          "<h2>Hệ thống Giám sát Thi cử Thông minh — Exam Monitor System</h2>"
          "<table class='meta'>"
          "<tr><td><b>Ngày giám sát:</b> %1</td><td><b>Phòng thi:</b> %2</td></tr>"
          "<tr><td><b>Giám thị coi thi:</b> %3</td><td><b>Tổng thời lượng:</b> %4 phút (%5 phiên)</td></tr>"
          "<tr><td><b>Tổng cảnh báo AI:</b> %6 sự cố</td>"
          "<td><b>Kết quả xác minh:</b> ✔ %7 vi phạm · ✖ %8 báo sai · ⏳ %9 chờ duyệt (Độ chính xác: %10)</td></tr>"
          "</table>"
          "<table class='inc'>"
          "<thead><tr>"
          "<th width='5%'>STT</th>"
          "<th width='16%'>Thời điểm / Phòng</th>"
          "<th width='10%'>Vị trí ghế</th>"
          "<th width='24%'>Hành vi & Chi tiết</th>"
          "<th width='10%'>Mức độ</th>"
          "<th width='15%'>Xác minh & Ghi chú</th>"
          "<th width='20%'>Ảnh bằng chứng</th>"
          "</tr></thead>"
          "<tbody>%11</tbody>"
          "</table>"
          "<table class='sig'>"
          "<tr>"
          "<td><b>GIÁM THỊ COI THI</b><br/><i>(Ký và ghi rõ họ tên)</i><br/><br/><br/><br/>%3</td>"
          "<td><b>ĐẠI DIỆN HỘI ĐỒNG THI</b><br/><i>(Ký và ghi rõ họ tên)</i><br/><br/><br/><br/>........................................</td>"
          "</tr>"
          "</table>"
          "<p style='color:#64748b;font-size:8pt;margin-top:20px;text-align:right;'>"
          "Biên bản được kết xuất lúc %12 từ Exam Monitor System.</p>"
          "</body></html>")
          .arg(day_.date.toString("dd/MM/yyyy"))
          .arg(rooms.isEmpty() ? QString("—") : rooms.join(", "))
          .arg(proctors.isEmpty() ? QString("—") : proctors.join(", "))
          .arg(total_sec / 60.0, 0, 'f', 1)
          .arg(day_.sessions.size())
          .arg(day_.incidents.size())
          .arg(n_confirmed)
          .arg(n_false)
          .arg(n_unverified)
          .arg(precision_str)
          .arg(rows_html)
          .arg(QDateTime::currentDateTime().toString("dd/MM/yyyy HH:mm:ss"));

  QPdfWriter pdf(pdfPath);
  pdf.setPageSize(QPageSize(QPageSize::A4));
  pdf.setPageOrientation(QPageLayout::Portrait);
  pdf.setResolution(300);
  pdf.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout::Millimeter);

  QTextDocument doc;
  doc.setHtml(html);
  doc.print(&pdf);

  QDesktopServices::openUrl(QUrl::fromLocalFile(pdfPath));
}
