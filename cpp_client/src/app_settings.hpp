#pragma once

#include <QCoreApplication>
#include <QDir>
#include <QSettings>
#include <QString>

// Cài đặt người dùng (lưu bằng QSettings — registry HKCU\Software\IUH\ExamMonitorSystem).
// Lần chạy đầu lấy mặc định từ biến môi trường trong scripts/local_env.bat.
struct AppSettings {
  int camera_index = 1;
  QString camera_source;   // tùy chọn: file video / rtsp:// / http:// (IP camera) — ưu tiên hơn camera_index
  QString exam_room = "PHONG_THI_01";
  QString proctor;
  int send_fps = 4;   // Số frame/giây gửi sang AI (AI phân tích 2 nhịp/giây)
  // Độ phân giải yêu cầu camera (chiều cao: 720 hoặc 1080). 1080p cho mặt người ở xa to hơn ~1,5 lần (Face Mesh,
  // cảm xúc); camera không hỗ trợ sẽ tự trả độ phân giải khác — xem dòng thông số dưới khung video
  int camera_height = 1080;
  // Nhận diện cảm xúc FER+ — mặc định BẬT, chỉ chạy cho chỗ có người và thấy mặt; chỉ tham khảo, không dùng kết luận gian lận
  bool enable_emotion = true;
  QString data_dir;   // Thư mục data/ mà AI ghi evidence/reports/logs
  QString sensitivity = "medium";     // Độ nhạy AI: "low" | "medium" | "high"
  int absent_alert_seconds = 15;      // Thời gian vắng mặt tối thiểu (giây) để cảnh báo "Rời chỗ" (15s kiểm thử, 60-300s thi thật)
  QString seatmap_path;               // Đường dẫn file seatmap.json cố định (để trống = tự động học sơ đồ ghế)

  QString video_label() const {
    return camera_source.isEmpty() ? QString("Camera %1").arg(camera_index) : camera_source;
  }

  static QString default_data_dir() {
    // Cùng biến môi trường với python_ai/main.py (--data-dir) để 2 bên luôn trỏ cùng 1 chỗ
    if (!qEnvironmentVariable("EXAM_DATA_DIR").isEmpty()) return qEnvironmentVariable("EXAM_DATA_DIR");
    // exe nằm trong build/ -> data/ ở thư mục gốc dự án
    QDir dir(QCoreApplication::applicationDirPath());
    if (dir.dirName().compare("build", Qt::CaseInsensitive) == 0) dir.cdUp();
    return dir.filePath("data");
  }

  static AppSettings load() {
    AppSettings s;
    bool ok = false;
    int env_cam = qEnvironmentVariableIntValue("EXAM_CAMERA_INDEX", &ok);
    if (ok) s.camera_index = env_cam;
    if (qEnvironmentVariableIsSet("EXAM_ROOM") && !qEnvironmentVariable("EXAM_ROOM").isEmpty())
      s.exam_room = qEnvironmentVariable("EXAM_ROOM");
    s.proctor = qEnvironmentVariable("EXAM_PROCTOR");
    s.camera_source = qEnvironmentVariable("EXAM_CAMERA_SOURCE");
    if (!qEnvironmentVariable("EXAM_SENSITIVITY").isEmpty())
      s.sensitivity = qEnvironmentVariable("EXAM_SENSITIVITY").toLower();
    s.data_dir = default_data_dir();

    QSettings q;
    s.camera_index = q.value("camera_index", s.camera_index).toInt();
    s.camera_source = q.value("camera_source", s.camera_source).toString();
    s.exam_room = q.value("exam_room", s.exam_room).toString();
    s.proctor = q.value("proctor", s.proctor).toString();
    s.send_fps = q.value("send_fps", s.send_fps).toInt();
    s.camera_height = q.value("camera_height", s.camera_height).toInt() >= 1080 ? 1080 : 720;
    s.enable_emotion = q.value("enable_emotion", s.enable_emotion).toBool();
    s.data_dir = q.value("data_dir", s.data_dir).toString();
    s.sensitivity = q.value("sensitivity", s.sensitivity).toString().toLower();
    if (s.sensitivity != "low" && s.sensitivity != "medium" && s.sensitivity != "high")
      s.sensitivity = "medium";
    s.absent_alert_seconds = qBound(5, q.value("absent_alert_seconds", s.absent_alert_seconds).toInt(), 600);
    s.seatmap_path = q.value("seatmap_path", s.seatmap_path).toString();
    return s;
  }

  void save() const {
    QSettings q;
    q.setValue("camera_index", camera_index);
    q.setValue("camera_source", camera_source);
    q.setValue("exam_room", exam_room);
    q.setValue("proctor", proctor);
    q.setValue("send_fps", send_fps);
    q.setValue("camera_height", camera_height);
    q.setValue("data_dir", data_dir);
    q.setValue("enable_emotion", enable_emotion);
    q.setValue("sensitivity", sensitivity);
    q.setValue("absent_alert_seconds", absent_alert_seconds);
    q.setValue("seatmap_path", seatmap_path);
  }
};
