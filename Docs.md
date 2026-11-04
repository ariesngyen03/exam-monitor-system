Author: DevAries

-- How to use Git-- 
1. git checkout main
2. git pull origin main
3. Create a branch git checkout -b [branch-name]
4. git add . 
5. git commit -m "message"


--- Smart Pointer in Cpp ---
- Khác với C truyền thống hay C++ cũ thì nó sẽ quyết được vấn đề là sẽ phải tự xóa bằng lệnh **delete**, nếu quên xóa thì RAM sẽ bị đầy dần cho đến khi sập máy (Memory Leak).
=> Nó sẽ giải quyết được vấn đề đó, và cực kì thông minh ở chỗ : Nó sẽ tự biết xem đang có bao nhieu người cầm tờ giấy ghi địa chỉ của nó. Khi tất cả mọi người đều đã xé bỏ tờ giấy địa chỉ (không ai dùng nữa), nó sẽ TỰ ĐỘNG dọn rác và giải phóng RAM