# 🛡️ Ứng dụng Phát Hiện Gian Lận & Dự Báo Rủi Ro Tín Dụng

Ứng dụng web tương tác này được phát triển bằng **Streamlit** dựa trên quy trình nghiên cứu, xây dựng và đánh giá mô hình học máy từ file Jupyter Notebook (`phat_hien_giao_dich_gian_lan.ipynb`). Ứng dụng tích hợp thuật toán phân loại mạnh mẽ để tự động hóa nghiệp vụ phát hiện giao dịch bất thường hoặc đánh giá khả năng nợ xấu (`default`) của khách hàng dựa trên các tập thuộc tính đầu vào.

## 🤖 Mô hình học máy sử dụng
- **Thuật toán chính:** `RandomForestClassifier` (Rừng ngẫu nhiên phân loại).
- **Mục tiêu dự báo:** Phân loại nhị phân lớp `default` (`0`: Hợp pháp/An toàn, `1`: Gian lận/Rủi ro cao).
- **Đặc trưng đầu vào:** Gồm 14 trường thông tin số liên tục đã định danh từ `X_1` đến `X_14`.

## 🛠️ Hướng dẫn cài đặt và Khởi chạy

### Bước 1: Chuẩn bị môi trường máy tính
Hãy đảm bảo bạn đã cài đặt sẵn Python (phiên bản khuyến nghị là từ `>= 3.9` đến `3.12`).

### Bước 2: Cài đặt các thư viện cần thiết
Mở terminal hoặc cửa sổ Command Prompt tại thư mục chứa mã nguồn này và chạy lệnh:
```bash
pip install -r requirements.txt
