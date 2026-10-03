# Career Lens

Ứng dụng phân tích dữ liệu tuyển dụng và khám phá xu hướng kỹ năng công nghệ. Dự án được xây dựng bằng Python và Streamlit, sử dụng dữ liệu việc làm trong `data/job.csv` để hiển thị các biểu đồ về vị trí, cấp độ, địa điểm và mức lương.

## Mục lục

- [Tính năng](#tính-năng)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu](#yêu-cầu)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Dữ liệu](#dữ-liệu)
- [Các mô-đun xử lý](#các-mô-đun-xử-lý)
- [Tình trạng các chức năng](#tình-trạng-các-chức-năng)
- [Khắc phục sự cố](#khắc-phục-sự-cố)

## Tính năng

### Dashboard

Dashboard đọc dữ liệu từ `data/job.csv` và cung cấp:

- Top 10 kỹ năng phổ biến, có thể sắp xếp theo tỷ lệ xuất hiện.
- Biểu đồ phân bố tin tuyển dụng theo cấp độ (`level`).
- Heatmap mức lương trung bình theo cấp độ và địa điểm; có thể chọn số địa điểm hiển thị.
- Bảng dữ liệu tương tác cho từng biểu đồ.

### Salary Predictor

Giao diện nhận CV PDF hoặc DOCX, gọi pipeline phân tích CV và sử dụng các artifacts mô hình để tạo dự đoán lương cùng thông tin tóm tắt. Cần có các artifacts trong `models/` để hoàn tất dự đoán. Xem [Tình trạng các chức năng](#tình-trạng-các-chức-năng) về khác biệt tên trường giữa kết quả parser và phần giao diện.

## Công nghệ sử dụng

- **Giao diện ứng dụng:** Streamlit, Streamlit Option Menu và các thành phần mở rộng Streamlit.
- **Phân tích dữ liệu:** pandas, NumPy.
- **Biểu đồ:** Plotly.
- **Mô hình học máy:** scikit-learn, LightGBM và joblib.
- **Đọc và phân tích CV:** PyMuPDF, python-docx và spaCy.
- **Giải thích mô hình:** SHAP và matplotlib trong mô-đun `core/SHAP_TreeExplainer.py`.

Danh sách thư viện khai báo trong `requirement.txt`.

## Cấu trúc dự án

```text
.
├── app.py                       # Điểm vào của ứng dụng Streamlit
├── requirement.txt              # Danh sách thư viện Python
├── assets/
│   └── style.css                # CSS tùy chỉnh cho giao diện
├── core/
│   ├── cv_reader.py             # Đọc, làm sạch và trích xuất thông tin từ CV
│   ├── salary_predictor.py      # Tiền xử lý dữ liệu và huấn luyện mô hình lương
│   ├── skill_listed.py          # Thống kê tỷ lệ kỹ năng từ dữ liệu việc làm
│   └── SHAP_TreeExplainer.py    # Phân tích/giải thích dự đoán bằng SHAP
├── data/
│   └── job.csv                  # Dữ liệu tin tuyển dụng đầu vào
├── pickle/                      # Artifacts mô hình cũ
└── docs/                        # Tài liệu thiết kế và kế hoạch phát triển
```

## Yêu cầu

- Python 3.10 trở lên được khuyến nghị; một số thư viện khoa học dữ liệu có thể cần phiên bản Python được hỗ trợ tương ứng.
- `pip` và môi trường ảo Python.
- Có file dữ liệu `data/job.csv` với các cột được ứng dụng sử dụng, tối thiểu gồm `job_title`, `level`, `location`, `avg_salary`; phần xử lý kỹ năng cũng sử dụng cột `skill` hoặc `skills`. Một số quy trình dự đoán còn cần `years_experience`.

## Cài đặt và chạy

Mở PowerShell hoặc terminal tại thư mục gốc của dự án.

1. Tạo và kích hoạt môi trường ảo:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Trên macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Cài thư viện:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirement.txt
   ```

3. Khởi chạy ứng dụng:

   ```bash
   streamlit run app.py
   ```

   Streamlit sẽ in địa chỉ ứng dụng trong terminal và thường tự mở trình duyệt. Dừng ứng dụng bằng `Ctrl+C`.

## Dữ liệu

Dashboard nạp dữ liệu tại đường dẫn tương đối với `app.py`: `data/job.csv`. Các trường đang được sử dụng gồm:

| Cột | Mục đích |
|---|---|
| `job_title` | Biến đầu vào cho quy trình huấn luyện mô hình |
| `level` | Thống kê cấp độ tuyển dụng |
| `location` | Lọc các địa điểm có nhiều tin tuyển dụng |
| `avg_salary` | Tính mức lương trung bình và tạo heatmap |
| `skills` | Danh sách kỹ năng, dùng cho thống kê và mô hình |
| `years_experience` | Số năm kinh nghiệm, dùng trong quy trình huấn luyện mô hình |

Một số cột chỉ được dùng bởi các mô-đun phụ trợ. Nếu thiếu cột mà một mô-đun yêu cầu, quy trình đó có thể báo lỗi dù Dashboard vẫn có thể chạy nếu các cột hiển thị biểu đồ còn đủ.

## Các mô-đun xử lý

- `core/cv_reader.py` có pipeline đọc PDF, DOCX, CSV và TXT, làm sạch văn bản, trích xuất một số thông tin liên hệ, kỹ năng và kinh nghiệm. Pipeline xử lý nhiều CV bằng `ProcessPoolExecutor`.
- `core/salary_predictor.py` tiền xử lý kỹ năng và biến phân loại, chia dữ liệu train/test, huấn luyện `LGBMRegressor` và lưu model cùng encoder/feature names vào thư mục `models/`.
- `core/skill_listed.py` tạo thống kê số lượng và tỷ lệ xuất hiện của kỹ năng từ `data/job.csv`, sau đó ghi kết quả vào `data/skill_listed.csv`.
- `core/SHAP_TreeExplainer.py` nạp các artifacts từ `models/` để hỗ trợ giải thích dự đoán. Tên và vị trí artifacts phải khớp với cấu hình trong mô-đun.

Các script trong `core/` là công cụ xử lý riêng; chạy ứng dụng bằng `streamlit run app.py` không tự động chạy toàn bộ pipeline huấn luyện hoặc phân tích CV.

## Tình trạng các chức năng

- Dashboard đọc `data/job.csv` và thống kê kỹ năng từ `data/skill_listed.csv`; nếu file thống kê chưa có, ứng dụng tạo nó từ dữ liệu việc làm.
- Giao diện Salary Predictor gọi `core/cv_reader.py` và sau đó nạp artifacts `mlb.pkl`, `gbm_model.pkl`, `feature_names.pkl` từ `models/`.
- Parser trả về các trường như `skills`, `years`, `level`, trong khi phần dự đoán trong `app.py` hiện đọc `skill`, `năm kinh nghiệm`, `level công việc`. Cần đồng bộ các tên trường để thông tin CV được đưa đầy đủ vào mô hình.
- `core/salary_predictor.py` đọc dữ liệu và bắt đầu huấn luyện ngay khi module được import. Chỉ chạy quy trình này khi dữ liệu đầu vào đã sẵn sàng và bạn chủ động muốn huấn luyện.