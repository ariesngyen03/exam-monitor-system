#pragma once

#include <QDate>
#include <QList>
#include <QString>
#include <QStringList>

// Đọc và cập nhật báo cáo phiên do AI (SessionReporter) ghi ra: <data_dir>/reports/YYYY-MM-DD.json
namespace reports {

struct Incident {
  int incident_id = 0;
  QString session_id;
  QString room;
  QString proctor;
  QString detected_at;   // "YYYY-MM-DD HH:MM:SS"
  QString seat_label;    // "D1-G2" hoặc "S1"
  QStringList violations;
  QString behavior_type; // suy ra từ violations (cùng quy tắc với exam_ai_service.classify_behavior)
  QString level;         // "THAP" | "TRUNG BINH" | "CAO" | "RAT CAO"
  QString assessment;
  QString screenshot_path;
  double duration_seconds = 0.0;
  QString verify_status = "unverified"; // "unverified" | "confirmed" | "false_alarm"
  QString proctor_note;
};

struct Session {
  QString session_id;
  QString room;
  QString proctor;
  QString start_time;
  QString end_time;
  double duration_seconds = 0.0;
  int incident_count = 0;
};

struct DayReport {
  QDate date;
  QList<Session> sessions;
  QList<Incident> incidents;
  QString error;   // rỗng nếu đọc thành công
};

QString reports_dir(const QString &data_dir);
QList<QDate> available_dates(const QString &data_dir);   // mới nhất trước
DayReport load_day(const QString &data_dir, const QDate &date);
QString behavior_from_violations(const QStringList &violations);
QString verify_status_display(const QString &status);
bool update_incident_verification(const QString &data_dir, const QDate &date,
                                  const QString &session_id, const QString &detected_at,
                                  const QString &seat_label, const QString &verify_status,
                                  const QString &proctor_note);

} // namespace reports
