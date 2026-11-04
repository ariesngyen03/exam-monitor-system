#include "live_page.hpp"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QStringList>
#include <algorithm>

#include "../src/behavior_labels.hpp"
#include "ui_helpers.hpp"

LivePage::LivePage(QWidget *parent) : QWidget(parent) {
  auto *root = new QHBoxLayout(this);
  root->setContentsMargins(24, 24, 24, 24);
  root->setSpacing(16);

  QVBoxLayout *l = nullptr;
  auto *video_card = ui::make_card(QString::fromUtf8("🔴 CAMERA TRỰC TIẾP"), &l);
  camera_slot_ = new QVBoxLayout();
  camera_slot_->setContentsMargins(0, 0, 0, 0);
  l->addLayout(camera_slot_, 1);
  root->addWidget(video_card, 3);

  auto *seat_card = ui::make_card(QString::fromUtf8("🪑 TRẠNG THÁI CHỖ NGỒI"), &l);
  seat_card->setMinimumWidth(460);
  summary_ = new QLabel(QString::fromUtf8("Chưa có dữ liệu từ AI"));
  summary_->setWordWrap(true);
  summary_->setStyleSheet("color: #475569;");
  table_ = new QTableView();
  table_->setObjectName("LogTableLight");
  table_->verticalHeader()->setVisible(false);
  table_->verticalHeader()->setDefaultSectionSize(34);
  table_->setSelectionMode(QAbstractItemView::NoSelection);
  table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_->setShowGrid(false);
  table_->setWordWrap(true);
  model_ = new QStandardItemModel(0, 5, this);
  model_->setHorizontalHeaderLabels({QString::fromUtf8("Chỗ"), QString::fromUtf8("Chú ý"),
                                     QString::fromUtf8("Trạng thái"), QString::fromUtf8("Mức độ"),
                                     QString::fromUtf8("Chi tiết")});
  table_->setModel(model_);
  auto *hh = table_->horizontalHeader();
  hh->setSectionResizeMode(QHeaderView::Fixed);
  hh->setSectionResizeMode(4, QHeaderView::Stretch);
  table_->setColumnWidth(0, 68);
  table_->setColumnWidth(1, 72);
  table_->setColumnWidth(2, 130);
  table_->setColumnWidth(3, 86);
  l->addWidget(summary_);
  l->addWidget(table_, 1);
  root->addWidget(seat_card, 2);
}

void LivePage::update_status(bool calibrating, float progress, int n_people, int n_faces) {
  // Khi đang giám sát, dòng tóm tắt do update_seats() viết (được gọi ngay trước hàm này)
  if (calibrating) {
    summary_->setText(QString::fromUtf8("<b>Đang học sơ đồ chỗ ngồi... %1%</b> — phát hiện %2 người, %3 khuôn mặt")
                          .arg(static_cast<int>(progress * 100.0f))
                          .arg(n_people)
                          .arg(n_faces));
  }
}

void LivePage::update_seats(const std::vector<exam::Detection> &detections) {
  std::vector<const exam::Detection *> seats;
  int present = 0, empty = 0, flagged = 0, others = 0;
  for (const auto &d : detections) {
    if (d.desk_id() <= 0) {   // Calibrating / Unassigned: không phải chỗ ngồi đã học
      others++;
      continue;
    }
    seats.push_back(&d);
    if (d.is_present()) present++; else empty++;
    if (d.is_cheating()) flagged++;
  }
  // Chỗ cần chú ý lên đầu, sau đó theo điểm chú ý của AI giảm dần (xếp hạng đáng tin hơn kết luận có/không)
  std::sort(seats.begin(), seats.end(), [](const exam::Detection *a, const exam::Detection *b) {
    if (a->is_cheating() != b->is_cheating()) return a->is_cheating();
    if (a->is_present() != b->is_present()) return a->is_present();
    return a->confidence() > b->confidence();
  });

  // Tái sử dụng các dòng đã có thay vì xóa và cấp phát mới mỗi nhịp (0.5s) để giảm tải cho UI thread
  model_->setRowCount(static_cast<int>(seats.size()));
  for (int r = 0; r < static_cast<int>(seats.size()); ++r) {
    const auto *d = seats[r];
    const std::string &type = d->behavior_type();
    QStringList detail;
    // Cảm xúc đứng đầu: cột Chi tiết hẹp, nối cuối danh sách vi phạm thì bị cắt mất
    if (!d->emotion().empty()) detail << QString::fromUtf8("Cảm xúc: %1").arg(QString::fromStdString(d->emotion()));
    for (const auto &v : d->violations()) detail << QString::fromStdString(v);
    // Điểm chú ý 0..1 -> thanh 5 nấc (không hiện %, đây là điểm xếp hạng chứ không phải xác suất)
    const int bars = d->is_present() ? std::clamp(static_cast<int>(d->confidence() * 5.0f + 0.5f), 0, 5) : 0;

    auto ensure_item = [this, r](int c, const QString &text) {
      QStandardItem *it = model_->item(r, c);
      if (!it) {
        it = new QStandardItem(text);
        model_->setItem(r, c, it);
      } else {
        it->setText(text);
      }
      return it;
    };

    ensure_item(0, QString::fromStdString(d->seat_label()));
    auto *attn = ensure_item(1, QString(bars, QChar(0x25AE)) + QString(5 - bars, QChar(0x25AF)));
    attn->setForeground(bars >= 4 ? QColor("#ef4444") : bars >= 3 ? QColor("#f59e0b") : QColor("#94a3b8"));
    attn->setToolTip(QString::fromUtf8("Điểm chú ý của AI — dùng để xếp hạng chỗ nên quan sát, không phải xác suất gian lận"));

    auto *st_item = ensure_item(2, behavior::display_name(type));
    st_item->setForeground(behavior::color(type));
    QFont f = st_item->font();
    f.setBold(d->is_cheating());
    st_item->setFont(f);

    ensure_item(3, behavior::risk_level_display(d->risk_level()));
    auto *det_item = ensure_item(4, detail.join("; "));
    det_item->setToolTip(detail.join("\n"));
  }

  if (!seats.empty() || others == 0) {
    summary_->setText(QString::fromUtf8("<b>%1</b> chỗ đã học · <b>%2</b> có mặt · <b>%3</b> trống · "
                                        "<span style='color:#ef4444;'><b>%4</b> cần chú ý</span>%5")
                          .arg(seats.size())
                          .arg(present)
                          .arg(empty)
                          .arg(flagged)
                          .arg(others > 0 ? QString::fromUtf8(" · %1 người chưa khớp chỗ").arg(others) : QString()));
  }
}
