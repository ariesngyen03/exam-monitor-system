#pragma once

#include <QColor>
#include <QRegularExpression>
#include <QString>
#include <string>

// Ánh xạ behavior_type (do python_ai/exam_ai_service.py sinh ra) sang tên tiếng Việt và màu hiển thị.
namespace behavior {

inline QString display_name(const std::string &type) {
  if (type == "Normal") return QString::fromUtf8("Bình thường");
  if (type == "Turning Head") return QString::fromUtf8("Quay đầu / Nhìn lệch");
  if (type == "Head Down") return QString::fromUtf8("Cúi đầu bất thường");
  if (type == "Leaning") return QString::fromUtf8("Nghiêng người");
  if (type == "Suspicious") return QString::fromUtf8("Hành vi bất thường");
  if (type == "Absent") return QString::fromUtf8("Rời chỗ");
  if (type == "Empty") return QString::fromUtf8("Chỗ trống");
  if (type == "Calibrating") return QString::fromUtf8("Đang học vị trí");
  if (type == "Unassigned") return QString::fromUtf8("Chưa xác định chỗ");
  return QString::fromStdString(type);
}

inline QColor color(const std::string &type) {
  if (type == "Normal") return QColor("#10b981");       // emerald-500
  if (type == "Turning Head") return QColor("#f59e0b"); // amber-500
  if (type == "Head Down") return QColor("#f97316");    // orange-500
  if (type == "Leaning") return QColor("#ef4444");      // red-500
  if (type == "Suspicious") return QColor("#e11d48");   // rose-600
  if (type == "Absent") return QColor("#a855f7");       // purple-500
  if (type == "Empty") return QColor("#94a3b8");        // slate-400
  return QColor("#3b82f6");                             // blue-500
}

// Hành vi cần cảnh báo (tô đỏ/cam trong bảng lịch sử)
inline bool is_violation(const std::string &type) {
  return type == "Turning Head" || type == "Head Down" || type == "Leaning" ||
         type == "Suspicious" || type == "Absent";
}

// "D1-G2" -> "Dãy 1 - Ghế 2"
inline QString seat_display(const std::string &label, int fallback_id) {
  static const QRegularExpression re("^D(\\d+)-G(\\d+)$");
  auto m = re.match(QString::fromStdString(label));
  if (m.hasMatch())
    return QString::fromUtf8("Dãy %1 - Ghế %2").arg(m.captured(1), m.captured(2));
  if (!label.empty()) return QString::fromStdString(label);
  return QString::fromUtf8("Khu vực %1").arg(fallback_id);
}

// "RAT CAO" -> "Rất cao"
inline QString risk_level_display(const std::string &level) {
  if (level == "RAT CAO") return QString::fromUtf8("Rất cao");
  if (level == "CAO") return QString::fromUtf8("Cao");
  if (level == "TRUNG BINH") return QString::fromUtf8("Trung bình");
  if (level == "THAP") return QString::fromUtf8("Thấp");
  return QString::fromStdString(level);
}

} // namespace behavior
