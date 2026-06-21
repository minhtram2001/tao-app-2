# Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán (Streamlit)

Ứng dụng web Streamlit giúp tối ưu hóa danh mục đầu tư chứng khoán HOSE dựa trên các chiến lược giao dịch kỹ thuật (RSI, RSI + MACD, RSI + Bollinger Bands) kết hợp thuật toán tối ưu hóa bầy đàn **Particle Swarm Optimization (PSO)** để tìm tham số tối ưu và phân bổ tỷ trọng theo phương pháp nghịch đảo biến động (**Inverse Volatility**).

---

## 🌟 Tính Năng Chính
1. **Tải Dữ Liệu Tự Do**: Hỗ trợ tải tệp CSV dữ liệu HOSE (chứa các cột `Date`, `Ticker`, `Open`, `Close` hoặc `adj_open`, `adj_close`).
2. **Cấu Hình Linh Hoạt**:
   - Vốn đầu tư ban đầu, phí giao dịch, ngưỡng cắt lỗ (Stop Loss).
   - Số lượng mã cổ phiếu nắm giữ mỗi kỳ ($N$).
   - Số lượng vòng lặp tối ưu hóa PSO (PSO Budget).
   - Chu kỳ tái cơ cấu danh mục: Hàng năm (Annual), Nửa năm (Semi-Annual), Hàng quý (Quarterly).
   - Phương thức phân bổ tỷ trọng: Chia đều (Equal) hoặc Nghịch đảo biến động (Inverse Volatility).
3. **Đánh Giá & So Sánh**:
   - So sánh hiệu suất của 3 chiến lược kết hợp (`RSI`, `RSI + MACD`, `RSI + Bollinger`) với chiến lược mua và nắm giữ toàn bộ thị trường (`Buy & Hold 100`).
   - Dashboard chỉ số tài chính chuyên sâu: Tỷ suất sinh lời (Return), CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Tỷ lệ phiên thắng.
4. **Trực Quan Hóa Sinh Động**:
   - Biểu đồ tăng trưởng tài sản danh mục đầu tư.
   - Biểu đồ lợi nhuận theo từng quý của chiến lược tốt nhất.
   - Biểu đồ cột so sánh trực quan các chỉ số Sharpe, Calmar giữa các chiến lược.
5. **Nhật Ký Tái Cơ Cấu & Giao Dịch Chi Tiết**:
   - Danh sách cổ phiếu được lựa chọn và tỷ trọng phân bổ từng kỳ.
   - Nhật ký toàn bộ các lệnh Mua/Bán (bao gồm cả lệnh Cắt lỗ) có thể tải xuống dưới dạng CSV.
6. **Kiểm Định Thống Kê**:
   - Thực hiện kiểm định t-test mẫu đơn (OOS Quarterly Return > 0) và kiểm định Wilcoxon signed-rank để xác định ý nghĩa thống kê của chiến lược.

---

## 🛠️ Hướng Dẫn Cài Đặt và Chạy Local

### 1. Chuẩn bị môi trường
Yêu cầu máy tính cài đặt **Python 3.8+**. 

Sao chép mã nguồn về máy tính và di chuyển vào thư mục dự án:
```bash
git clone <link-github-cua-ban>
cd <ten-thu-muc-du-an>
```

### 2. Cài đặt các thư viện cần thiết
Cài đặt các gói phụ thuộc từ tệp `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Streamlit
Khởi chạy ứng dụng Streamlit:
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Deploy Lên Streamlit Community Cloud

Streamlit cung cấp nền tảng hosting miễn phí cho các ứng dụng mã nguồn mở. Hãy thực hiện theo các bước sau để deploy:

### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một repository mới trên GitHub (ví dụ: `hose-portfolio-optimization`).
2. Đẩy các file `app.py`, `requirements.txt`, `README.md` lên repository đó.
   *(Lưu ý: Bạn có thể chọn tải tệp dữ liệu `HOSE_2020_2023.csv` lên GitHub nếu dung lượng nhỏ hơn 25MB để làm dữ liệu mặc định, hoặc người dùng sẽ tự tải lên giao diện ứng dụng).*

### Bước 2: Kết nối với Streamlit Community Cloud
1. Truy cập [Streamlit Share](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"**.

### Bước 3: Cấu hình deploy ứng dụng
Điền các thông tin của repository chứa dự án của bạn:
- **Repository**: `username/hose-portfolio-optimization` (đường dẫn tài khoản github của bạn)
- **Branch**: `main` (hoặc `master`)
- **Main file path**: `app.py`
- Nhấn **Deploy!**

Hệ thống Streamlit sẽ tự động tạo môi trường ảo, cài đặt các thư viện trong `requirements.txt` và khởi chạy web app. Sau vài phút, ứng dụng của bạn sẽ hoạt động trực tuyến với đường dẫn chia sẻ công khai!

---

## 📂 Cấu Trúc File Dự Án
- `app.py`: Code ứng dụng chính chứa giao diện Streamlit, công cụ backtest và thuật toán PSO.
- `requirements.txt`: Chứa danh sách các thư viện cần cài đặt để ứng dụng chạy.
- `README.md`: Hướng dẫn sử dụng và cài đặt ứng dụng (file này).
- `industry_tickers.py`: Định nghĩa danh sách các cổ phiếu theo ngành (sử dụng cho việc phân loại/lọc cổ phiếu nếu cần).
