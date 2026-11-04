#pragma once

#include <QColor>
#include <QFrame>
#include <QGraphicsDropShadowEffect>
#include <QLabel>
#include <QList>
#include <QPainter>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>
#include <algorithm>

namespace ui {

// Thẻ trắng bo góc có tiêu đề (đồng bộ với style.qss: QFrame#Card, QLabel#CardTitle)
inline QFrame *make_card(const QString &title, QVBoxLayout **body, QWidget *parent = nullptr) {
  auto *card = new QFrame(parent);
  card->setObjectName("Card");
  auto *shadow = new QGraphicsDropShadowEffect(card);
  shadow->setBlurRadius(15);
  shadow->setColor(QColor(0, 0, 0, 20));
  shadow->setOffset(0, 4);
  card->setGraphicsEffect(shadow);

  auto *layout = new QVBoxLayout(card);
  layout->setContentsMargins(20, 20, 20, 20);
  layout->setSpacing(12);
  if (!title.isEmpty()) {
    auto *lbl = new QLabel(title);
    lbl->setObjectName("CardTitle");
    layout->addWidget(lbl);
  }
  if (body) *body = layout;
  return card;
}

// Ô số liệu: nhãn nhỏ + giá trị lớn
inline QFrame *make_stat_tile(const QString &caption, QLabel **value_label, const QString &color,
                              QWidget *parent = nullptr) {
  auto *tile = new QFrame(parent);
  tile->setObjectName("StatTile");
  auto *l = new QVBoxLayout(tile);
  l->setContentsMargins(16, 12, 16, 12);
  auto *cap = new QLabel(caption);
  cap->setStyleSheet("color: #64748b; font-weight: 500; font-size: 12px; border: none;");
  auto *val = new QLabel("—");
  val->setStyleSheet(QString("color: %1; font-size: 26px; font-weight: bold; border: none;").arg(color));
  l->addWidget(cap);
  l->addWidget(val);
  if (value_label) *value_label = val;
  return tile;
}

// Biểu đồ cột ngang đơn giản (không cần Qt Charts)
class BarChart : public QWidget {
public:
  struct Bar {
    QString label;
    int value;
    QColor color;
  };

  explicit BarChart(QWidget *parent = nullptr) : QWidget(parent) { setMinimumHeight(60); }

  void set_bars(const QList<Bar> &bars) {
    bars_ = bars;
    setMinimumHeight(std::max(60, static_cast<int>(bars_.size()) * kRowH + 8));
    update();
  }

protected:
  void paintEvent(QPaintEvent *) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    if (bars_.isEmpty()) {
      p.setPen(QColor("#94a3b8"));
      p.drawText(rect(), Qt::AlignCenter, QString::fromUtf8("Chưa có dữ liệu"));
      return;
    }
    int max_v = 1;
    for (const auto &b : bars_) max_v = std::max(max_v, b.value);
    const int label_w = 150, value_w = 40;
    const int bar_area = std::max(10, width() - label_w - value_w - 16);
    QFont f = font();
    f.setPointSize(9);
    p.setFont(f);
    for (int i = 0; i < bars_.size(); ++i) {
      const auto &b = bars_[i];
      const int y = i * kRowH + 4;
      p.setPen(QColor("#475569"));
      p.drawText(QRect(0, y, label_w - 8, kRowH - 8), Qt::AlignVCenter | Qt::AlignRight,
                 p.fontMetrics().elidedText(b.label, Qt::ElideRight, label_w - 8));
      QRect track(label_w, y + 4, bar_area, kRowH - 16);
      p.setPen(Qt::NoPen);
      p.setBrush(QColor("#f1f5f9"));
      p.drawRoundedRect(track, 4, 4);
      QRect fill = track;
      fill.setWidth(std::max(2, bar_area * b.value / max_v));
      p.setBrush(b.color);
      p.drawRoundedRect(fill, 4, 4);
      p.setPen(QColor("#1e293b"));
      p.drawText(QRect(label_w + bar_area + 8, y, value_w, kRowH - 8), Qt::AlignVCenter | Qt::AlignLeft,
                 QString::number(b.value));
    }
  }

private:
  static constexpr int kRowH = 30;
  QList<Bar> bars_;
};

} // namespace ui
