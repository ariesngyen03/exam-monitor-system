/*Author: devAries*/

#include "../ui/main_window.hpp"
#include "app_settings.hpp"
#include "exam_monitor.pb.h"
#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QMetaType>
#include <QStyleHints>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>

// Khai báo để đăng ký với Qt
Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(exam::InferenceResult)

namespace {
// Ghi mọi dòng std::cout/std::cerr ra cả console lẫn file. Bản đóng gói chạy trên máy khác, không ai đọc cửa sổ
// console -> log (định dạng camera thật, lỗi mở camera...) phải nằm trong data/logs để gửi về cùng thư mục data.
// Có khoá dùng chung: cout/cerr được ghi từ nhiều luồng (camera, ZeroMQ, giao diện) vào cùng 1 file.
std::mutex g_log_mutex;

class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *console, std::streambuf *file) : console_(console), file_(file) {}

protected:
  int overflow(int c) override {
    if (c == traits_type::eof()) return traits_type::not_eof(c);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (console_) console_->sputc(static_cast<char>(c));
    file_->sputc(static_cast<char>(c));
    if (c == '\n') file_->pubsync();
    return c;
  }
  std::streamsize xsputn(const char *s, std::streamsize n) override {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (console_) console_->sputn(s, n);
    return file_->sputn(s, n);
  }
  int sync() override {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (console_) console_->pubsync();
    return file_->pubsync();
  }

private:
  std::streambuf *console_;
  std::streambuf *file_;
};
}  // namespace

int main(int argc, char *argv[]) {
  // Xác nhận Protobuf library version
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  QApplication app(argc, argv);
  app.setApplicationName("ExamMonitorSystem");
  app.setOrganizationName("IUH");
  // Giao diện thiết kế cho nền sáng: không theo dark mode của Windows
  // (nếu theo, chữ mặc định chuyển sang trắng và biến mất trên thẻ trắng)
  QGuiApplication::styleHints()->setColorScheme(Qt::ColorScheme::Light);
  app.setStyle("Fusion");

  // data/logs/app_console.log (cùng thư mục với ai_console.log của AI)
  const QString log_dir = QDir(AppSettings::load().data_dir).filePath("logs");
  QDir().mkpath(log_dir);
  std::ofstream log_file(std::filesystem::path(QDir(log_dir).filePath("app_console.log").toStdWString()),
                         std::ios::app);
  std::streambuf *cout_orig = std::cout.rdbuf();
  std::streambuf *cerr_orig = std::cerr.rdbuf();
  TeeBuf tee_out(cout_orig, log_file.rdbuf());
  TeeBuf tee_err(cerr_orig, log_file.rdbuf());
  if (log_file) {
    std::cout.rdbuf(&tee_out);
    std::cerr.rdbuf(&tee_err);
  }

  // Đăng ký meta-type để dùng trong Signal/Slot qua các Thread
  qRegisterMetaType<cv::Mat>("cv::Mat");
  qRegisterMetaType<exam::InferenceResult>("exam::InferenceResult");

  std::cout << "[SYSTEM] Exam Monitor System khoi dong "
            << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss").toStdString() << std::endl;

  int result = 0;
  {
    // Khối riêng: cửa sổ (và các luồng camera/ZeroMQ còn ghi log) huỷ xong trước khi trả lại cout/cerr gốc
    MainWindow window;
    window.show();
    result = app.exec();
  }

  // Giải phóng Protobuf
  google::protobuf::ShutdownProtobufLibrary();

  std::cout << "[SYSTEM] Exam Monitor System ket thuc." << std::endl;
  std::cout.rdbuf(cout_orig);
  std::cerr.rdbuf(cerr_orig);
  return result;
}