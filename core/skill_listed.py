import pandas as pd
import os
import sys

# Thêm đường dẫn thư mục gốc vào sys.path để import dễ dàng
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.salary_predictor import process_skills, encode_skills

def generate_skill_percentage_df() -> pd.DataFrame:
    """
    Hàm đọc dữ liệu job.csv, gọi hàm tạo list skill từ salary_predictor,
    sau đó tính toán phần trăm các skill chiếm trong dataset và xuất ra file CSV.
    """
    data_path = os.path.join(BASE_DIR, "data", "job.csv")
    output_path = os.path.join(BASE_DIR, "data", "skill_listed.csv")
    
    try:
        # Đọc dữ liệu từ file csv
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Lỗi khi đọc file dữ liệu: {e}")
        return pd.DataFrame()
        
    # Gọi hàm xử lý chuỗi thành list các skill
    df = process_skills(df)
    
    # Mã hóa các skill bằng MultiLabelBinarizer để đếm số lần xuất hiện
    skill_encoded, mlb = encode_skills(df)
    
    # Tính tổng số lần xuất hiện của từng kỹ năng
    skill_counts = skill_encoded.sum(axis=0)
    total_jobs = len(df)
    
    # Tạo DataFrame thống kê
    skill_df = pd.DataFrame({
        'Skill': mlb.classes_,
        'Count': skill_counts,
        'Percentage (%)': (skill_counts / total_jobs) * 100
    })
    
    # Sắp xếp theo phần trăm giảm dần
    skill_df = skill_df.sort_values(by='Percentage (%)', ascending=False).reset_index(drop=True)
    
    # Lưu DataFrame ra tệp csv
    try:
        skill_df.to_csv(output_path, index=False)
        print(f"Đã lưu danh sách kỹ năng thành công tại: {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file CSV: {e}")
        
    return skill_df

if __name__ == "__main__":
    generate_skill_percentage_df()
