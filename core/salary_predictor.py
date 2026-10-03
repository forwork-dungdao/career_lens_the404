import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "job.csv")

df = pd.read_csv(DATA_PATH)

def process_skills(df):
    col_name = 'skill' if 'skill' in df.columns else 'skills'
    df['skill_list'] = df[col_name].apply(
        lambda x: [s.strip() for s in str(x).split(',')] if pd.notnull(x) else []
    )
    return df

df_new = process_skills(df)
#print(df_new.head())
df = df_new
#dinh nghia tep muc tieu y
y = df["avg_salary"]

#ham encoding du lieu
def encode_skills(df):
    mlb = MultiLabelBinarizer()
    skill_encoded = mlb.fit_transform(df['skill_list'])
    return skill_encoded, mlb

skill_encoded, mlb = encode_skills(df)

def encode_categorical(df):
    return pd.get_dummies(df[["job_title", "level", "location"]])

categorical_col = encode_categorical(df)
numeric_col = df[["years_experience"]]

def final_features(skill_encoded, mlb, categorical_col, numeric_col):
    skill_df = pd.DataFrame(skill_encoded, columns=mlb.classes_, index=categorical_col.index)
    X = pd.concat([skill_df, categorical_col, numeric_col], axis=1)
    return X

X = final_features(skill_encoded, mlb, categorical_col, numeric_col)
import re
# Xử lý tên cột (LightGBM không hỗ trợ ký tự đặc biệt) và đảm bảo tính duy nhất
new_cols = []
for col in X.columns:
    clean = re.sub(r'[^A-Za-z0-9_]+', '_', str(col))
    if clean in new_cols:
        i = 1
        while f"{clean}_{i}" in new_cols:
            i += 1
        clean = f"{clean}_{i}"
    new_cols.append(clean)
X.columns = new_cols

# Chia tập train/test (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Khởi tạo mô hình LightGBM Regressor
gbm = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

# Huấn luyện mô hình với Early Stopping
gbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Đánh giá mô hình
y_pred = gbm.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Lưu mô hình theo quy tắc hệ thống
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(gbm, os.path.join(models_dir, "gbm_model.pkl"))
joblib.dump(mlb, os.path.join(models_dir, "mlb.pkl"))
joblib.dump(list(X.columns), os.path.join(models_dir, "feature_names.pkl"))

print(f"Đã lưu các artifacts vào: {models_dir}")
