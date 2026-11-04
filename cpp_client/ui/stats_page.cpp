#include "stats_page.hpp"

#include <QGridLayout>
#include <QHBoxLayout>
#include <QMap>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>
#include <algorithm>

#include "../src/behavior_labels.hpp"
#include "../src/report_store.hpp"

StatsPage::StatsPage(QWidget *parent) : QWidget(parent) {
  auto *outer = new QVBoxLayout(this);
  outer->setContentsMargins(0, 0, 0, 0);
  auto *scroll = new QScrollArea();
  scroll->setWidgetResizable(true);
  scroll->setFrameShape(QFrame::NoFrame);
  auto *content = new QWidget();
  scroll->setWidget(content);
  outer->addWidget(scroll);

  auto *root = new QVBoxLayout(content);
  root->setContentsMargins(24, 24, 24, 24);
  root->setSpacing(16);

  auto *bar = new QHBoxLayout();
  auto *lbl = new QLabel(QString::fromUtf8("Ngày:"));
  lbl->setStyleSheet("font-weight: 600; color: #334155;");
  date_combo_ = new QComboBox();
  date_combo_->setMinimumWidth(160);
  auto *btn_reload = new QPushButton(QString::fromUtf8("Làm mới"));
  btn_reload->setObjectName("BtnSecondary");
  bar->addWidget(lbl);
  bar->addWidget(date_combo_);
  bar->addWidget(btn_reload);
  bar->addStretch();
  root->addLayout(bar);

  // Ô số liệu tổng quan
  auto *tiles = new QHBoxLayout();
  tiles->setSpacing(16);
  tiles->addWidget(ui::make_stat_tile(QString::fromUtf8("Phiên giám sát"), &val_sessions_, "#3b82f6"));
  tiles->addWidget(ui::make_stat_tile(QString::fromUtf8("Sự cố ghi nhận"), &val_incidents_, "#ef4444"));
  tiles->addWidget(ui::make_stat_tile(QString::fromUtf8("Thời gian giám sát"), &val_hours_, "#10b981"));
  tiles->addWidget(ui::make_stat_tile(QString::fromUtf8("Sự cố / giờ giám sát"), &val_rate_, "#f59e0b"));
  tiles->addWidget(ui::make_stat_tile(QString::fromUtf8("Độ chính xác (Đã duyệt)"), &val_precision_, "#8b5cf6"));
  root->addLayout(tiles);

  auto *grid = new QGridLayout();
  grid->setSpacing(16);
  QVBoxLayout *l = nullptr;
  auto *c1 = ui::make_card(QString::fromUtf8("THEO HÀNH VI"), &l);
  chart_behavior_ = new ui::BarChart();
  l->addWidget(chart_behavior_);
  l->addStretch();
  auto *c2 = ui::make_card(QString::fromUtf8("THEO MỨC ĐỘ"), &l);
  chart_level_ = new ui::BarChart();
  l->addWidget(chart_level_);
  l->addStretch();
  auto *c3 = ui::make_card(QString::fromUtf8("THEO CHỖ NGỒI (10 chỗ nhiều nhất)"), &l);
  chart_seat_ = new ui::BarChart();
  l->addWidget(chart_seat_);
  l->addStretch();
  auto *c4 = ui::make_card(QString::fromUtf8("PHIÊN ĐANG CHẠY"), &l);
  live_caption_ = new QLabel();
  live_caption_->setStyleSheet("color: #64748b;");
  chart_live_ = new ui::BarChart();
  l->addWidget(live_caption_);
  l->addWidget(chart_live_);
  l->addStretch();
  grid->addWidget(c1, 0, 0);
  grid->addWidget(c2, 0, 1);
  grid->addWidget(c3, 1, 0);
  grid->addWidget(c4, 1, 1);
  root->addLayout(grid);

  auto *note = new QLabel(QString::fromUtf8(
      "Mức độ là đánh giá định tính từ luật hành vi + SeatScorer, không phải xác suất gian lận. "
      "Các cảnh báo đã được giám thị đánh dấu '✖ Báo sai' ở trang Lịch sử sẽ được loại trừ khỏi biểu đồ thống kê."));
  note->setWordWrap(true);
  note->setStyleSheet("color: #94a3b8; font-size: 12px;");
  root->addWidget(note);
  root->addStretch();

  connect(btn_reload, &QPushButton::clicked, this, &StatsPage::reload);
  connect(date_combo_, &QComboBox::currentIndexChanged, this, [this](int) { load_selected_date(); });
  refresh_live();
}

void StatsPage::set_data_dir(const QString &data_dir) {
  data_dir_ = data_dir;
  reload();
}

void StatsPage::reload() {
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

void StatsPage::load_selected_date() {
  const QDate date = date_combo_->currentData().toDate();
  reports::DayReport day;
  if (date.isValid()) day = reports::load_day(data_dir_, date);

  double total_sec = 0.0;
  for (const auto &s : day.sessions) total_sec += s.duration_seconds;
  const double hours = total_sec / 3600.0;
  val_sessions_->setText(QString::number(day.sessions.size()));
  val_incidents_->setText(QString::number(day.incidents.size()));
  val_hours_->setText(total_sec < 3600 ? QString::fromUtf8("%1 phút").arg(total_sec / 60.0, 0, 'f', 0)
                                       : QString::fromUtf8("%1 giờ").arg(hours, 0, 'f', 1));
  val_rate_->setText(hours > 0 ? QString::number(day.incidents.size() / hours, 'f', 1) : "—");

  int n_confirmed = 0, n_false = 0;
  QMap<QString, int> by_behavior, by_level, by_seat;
  for (const auto &inc : day.incidents) {
    if (inc.verify_status == "confirmed") {
      n_confirmed++;
    } else if (inc.verify_status == "false_alarm") {
      n_false++;
      continue;
    }
    by_behavior[inc.behavior_type]++;
    by_level[inc.level]++;
    by_seat[inc.seat_label]++;
  }
  const int n_verified = n_confirmed + n_false;
  val_precision_->setText(
      n_verified > 0
          ? QString("%1% (%2/%3)").arg(100.0 * n_confirmed / n_verified, 0, 'f', 0).arg(n_confirmed).arg(n_verified)
          : QString("—"));

  QList<ui::BarChart::Bar> bars;
  for (auto it = by_behavior.begin(); it != by_behavior.end(); ++it)
    bars << ui::BarChart::Bar{behavior::display_name(it.key().toStdString()), it.value(),
                              behavior::color(it.key().toStdString())};
  std::sort(bars.begin(), bars.end(), [](const auto &a, const auto &b) { return a.value > b.value; });
  chart_behavior_->set_bars(bars);

  bars.clear();
  const QList<QPair<QString, QString>> levels = {
      {"RAT CAO", "#b91c1c"}, {"CAO", "#ef4444"}, {"TRUNG BINH", "#f59e0b"}, {"THAP", "#facc15"}};
  for (const auto &lv : levels)
    if (by_level.contains(lv.first))
      bars << ui::BarChart::Bar{behavior::risk_level_display(lv.first.toStdString()), by_level[lv.first],
                                QColor(lv.second)};
  chart_level_->set_bars(bars);

  bars.clear();
  for (auto it = by_seat.begin(); it != by_seat.end(); ++it)
    bars << ui::BarChart::Bar{behavior::seat_display(it.key().toStdString(), 0), it.value(), QColor("#6366f1")};
  std::sort(bars.begin(), bars.end(), [](const auto &a, const auto &b) { return a.value > b.value; });
  if (bars.size() > 10) bars = bars.mid(0, 10);
  chart_seat_->set_bars(bars);
}

void StatsPage::add_live_alert(const QString &behavior_type) {
  live_counts_[behavior_type]++;
  refresh_live();
}

void StatsPage::reset_live() {
  live_counts_.clear();
  refresh_live();
}

void StatsPage::refresh_live() {
  int total = 0;
  QList<ui::BarChart::Bar> bars;
  for (auto it = live_counts_.begin(); it != live_counts_.end(); ++it) {
    total += it.value();
    bars << ui::BarChart::Bar{behavior::display_name(it.key().toStdString()), it.value(),
                              behavior::color(it.key().toStdString())};
  }
  std::sort(bars.begin(), bars.end(), [](const auto &a, const auto &b) { return a.value > b.value; });
  chart_live_->set_bars(bars);
  live_caption_->setText(QString::fromUtf8("%1 cảnh báo trong phiên hiện tại").arg(total));
}
