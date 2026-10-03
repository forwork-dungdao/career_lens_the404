import joblib
import pandas as pd
import os

def print_learned_skills():
    # Lấy đường dẫn tới thư mục models/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_names_path = os.path.join(base_dir, 'models', 'feature_names.pkl')
    
    try:
        # Load mảng chứa tên tất cả các features (cột)
        feature_names = joblib.load(feature_names_path)
        
        # Mảng feature_names chứa cả skill và các cột category (job_title_..., level_..., location_...) và years_experience
        # Ta tiến hành lọc chỉ lấy các skills (không có tiền tố của categorical/numeric)
        non_skill_prefixes = ('job_title_', 'level_', 'location_', 'years_experience')
        skills = [feat for feat in feature_names if not feat.startswith(non_skill_prefixes)]
        
        # Tạo DataFrame để in ra hiển thị dạng bảng
        df_skills = pd.DataFrame(skills, columns=["Kỹ Năng Đã Học (Learned Skills)"])
        
        print(f"✅ Đã tìm thấy file feature_names.pkl")
        print(f"🎯 Tổng số kỹ năng mô hình đã học: {len(df_skills)} kỹ năng.")
        
        # Để in ra toàn bộ bảng, bỏ giới hạn dòng của pandas
        pd.set_option('display.max_rows', None)
        print("\n--- BẢNG KỸ NĂNG MÔ HÌNH ĐÃ HỌC ---")
        print(df_skills)
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy tệp {feature_names_path}. Hãy chắc chắn rằng bạn đã train model và lưu thư mục models/")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    print_learned_skills()
