# Team Workflow

## First Principle

**Garbage In = Garbage Out**

Code đưa vào `main` phải được kiểm tra và review trước khi merge.

## B1. Clone repository

Mỗi thành viên clone project:

```bash
git clone https://github.com/hieungyu/multimodal-behavior-analysis.git
```

Sau khi clone sẽ có toàn bộ source code và cấu trúc dự án hiện tại.

## B2. Cập nhật main

Trước khi bắt đầu làm việc:

```bash
git checkout main
git pull origin main
```

Đảm bảo luôn làm việc trên phiên bản mới nhất.

## B3. Tạo branch cho từng task

Ví dụ:

```bash
git checkout -b feature/emotion-detection
```

hoặc

```bash
git checkout -b feature/ui-dashboard
```

Mỗi task hoặc tính năng mới phải được phát triển trên branch riêng.

## B4. Phát triển và commit

Trong quá trình làm việc:

```bash
git add .
git commit -m "message"
```

Commit thường xuyên với nội dung rõ ràng và dễ hiểu.

## B5. Push branch lên GitHub

```bash
git push origin feature/emotion-detection
```

## B6. Tạo Pull Request

Sau khi hoàn thành tính năng:

* Tạo Pull Request từ branch hiện tại vào `main`.
* Thành viên còn lại thực hiện review.

Kiểm tra:

* Logic xử lý
* Chất lượng code
* Kiến trúc hệ thống
* Nguy cơ phát sinh bug

## B7. Merge sau khi được chấp thuận

Chỉ merge khi cả hai đã thống nhất rằng thay đổi là an toàn và đạt yêu cầu.

## B8. Đồng bộ lại môi trường làm việc

Sau khi merge:

```bash
git checkout main
git pull origin main
```

Tiếp tục tạo branch mới cho nhiệm vụ tiếp theo.

## Team Rules

1. Không push trực tiếp lên `main`.
2. Mỗi tính năng hoặc nhiệm vụ phải có branch riêng.
3. Mọi thay đổi phải đi qua Pull Request.
4. Commit message phải rõ ràng và mô tả đúng nội dung thay đổi.
5. Review trước khi merge.
6. Luôn cập nhật `main` trước khi bắt đầu một nhiệm vụ mới.
