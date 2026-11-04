#include "report_store.hpp"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <algorithm>

namespace reports {

QString reports_dir(const QString &data_dir) { return QDir(data_dir).filePath("reports"); }

QList<QDate> available_dates(const QString &data_dir) {
  QList<QDate> dates;
  static const QRegularExpression re("^(\\d{4}-\\d{2}-\\d{2})\\.json$");
  const auto files = QDir(reports_dir(data_dir)).entryList({"*.json"}, QDir::Files);
  for (const QString &f : files) {
    auto m = re.match(f);
    if (!m.hasMatch()) continue;
    QDate d = QDate::fromString(m.captured(1), "yyyy-MM-dd");
    if (d.isValid()) dates << d;
  }
  std::sort(dates.begin(), dates.end(), std::greater<QDate>());
  return dates;
}

QString behavior_from_violations(const QStringList &violations) {
  // Cùng thứ tự ưu tiên với python_ai/exam_ai_service.py::_BEHAVIOR_PRIORITY
  static const QList<QPair<QString, QString>> priority = {
      {"sang cho khac", "Leaning"}, {"quay dau", "Turning Head"},
      {"nhin lech", "Turning Head"}, {"cui/ngua dau", "Head Down"},
      {"nghieng nguoi", "Leaning"},  {"roi cho", "Absent"},
  };
  const QString joined = violations.join(' ').toLower();
  for (const auto &p : priority)
    if (joined.contains(p.first)) return p.second;
  return "Suspicious";
}

QString verify_status_display(const QString &status) {
  if (status == "confirmed") return QString::fromUtf8("✔ Vi phạm thật");
  if (status == "false_alarm") return QString::fromUtf8("✖ Báo sai");
  return QString::fromUtf8("⏳ Chờ duyệt");
}

DayReport load_day(const QString &data_dir, const QDate &date) {
  DayReport day;
  day.date = date;
  QFile f(QDir(reports_dir(data_dir)).filePath(date.toString("yyyy-MM-dd") + ".json"));
  if (!f.open(QIODevice::ReadOnly)) {
    day.error = QString::fromUtf8("Không có báo cáo cho ngày %1").arg(date.toString("dd/MM/yyyy"));
    return day;
  }
  QJsonParseError err{};
  const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
  if (err.error != QJsonParseError::NoError || !doc.isObject()) {
    day.error = QString::fromUtf8("File báo cáo lỗi: %1").arg(err.errorString());
    return day;
  }

  for (const QJsonValue &sv : doc.object().value("sessions").toArray()) {
    const QJsonObject so = sv.toObject();
    Session s;
    s.session_id = so.value("session_id").toString();
    s.room = so.value("exam_room").toString();
    s.proctor = so.value("proctor").toString();
    s.start_time = so.value("start_time").toString();
    s.end_time = so.value("end_time").toString();
    s.duration_seconds = so.value("duration_seconds").toDouble();

    for (const QJsonValue &iv : so.value("incidents").toArray()) {
      const QJsonObject io = iv.toObject();
      Incident inc;
      inc.incident_id = io.value("incident_id").toInt(0);
      inc.session_id = s.session_id;
      inc.room = s.room;
      inc.proctor = s.proctor;
      inc.detected_at = io.value("detected_at").toString();
      inc.seat_label = io.value("seat_label").toString();
      for (const QJsonValue &v : io.value("violations").toArray()) inc.violations << v.toString();
      // Báo cáo mới lưu sẵn behavior_type lúc mở sự cố; báo cáo cũ thì suy ra từ violations
      inc.behavior_type = io.value("behavior_type").toString();
      if (inc.behavior_type.isEmpty()) inc.behavior_type = behavior_from_violations(inc.violations);
      inc.level = io.value("confidence_level").toString();
      inc.assessment = io.value("assessment").toString();
      inc.screenshot_path = io.value("screenshot_path").toString();
      inc.duration_seconds = io.value("duration_seconds").toDouble();
      inc.verify_status = io.value("verify_status").toString("unverified");
      if (inc.verify_status.isEmpty()) inc.verify_status = "unverified";
      inc.proctor_note = io.value("proctor_note").toString();
      day.incidents << inc;
      s.incident_count++;
    }
    day.sessions << s;
  }
  // Mới nhất lên đầu
  std::sort(day.incidents.begin(), day.incidents.end(),
            [](const Incident &a, const Incident &b) { return a.detected_at > b.detected_at; });
  return day;
}

bool update_incident_verification(const QString &data_dir, const QDate &date,
                                  const QString &session_id, const QString &detected_at,
                                  const QString &seat_label, const QString &verify_status,
                                  const QString &proctor_note) {
  const QString path = QDir(reports_dir(data_dir)).filePath(date.toString("yyyy-MM-dd") + ".json");
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly)) return false;

  QJsonParseError err{};
  QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
  f.close();
  if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

  QJsonObject root = doc.object();
  QJsonArray sessions = root.value("sessions").toArray();
  bool updated = false;

  for (int i = 0; i < sessions.size(); ++i) {
    QJsonObject so = sessions[i].toObject();
    if (!session_id.isEmpty() && so.value("session_id").toString() != session_id) continue;

    QJsonArray incidents = so.value("incidents").toArray();
    for (int j = 0; j < incidents.size(); ++j) {
      QJsonObject io = incidents[j].toObject();
      if (io.value("detected_at").toString() == detected_at &&
          io.value("seat_label").toString() == seat_label) {
        io["verify_status"] = verify_status;
        io["proctor_note"] = proctor_note;
        incidents[j] = io;
        updated = true;
        break;
      }
    }
    if (updated) {
      so["incidents"] = incidents;
      sessions[i] = so;
      break;
    }
  }

  if (!updated) return false;
  root["sessions"] = sessions;

  if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
  f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  f.close();
  return true;
}

} // namespace reports
