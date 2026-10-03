import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

def process_skills(df):
    col_name = 'skill' if 'skill' in df.columns else 'skills'
    df['skill_list'] = df[col_name].apply(
        lambda x: [s.strip() for s in str(x).split(',')] if pd.notnull(x) else []
    )
    return df

df_new = pd.read_csv("data/job.csv")
df_new = process_skills(df_new)
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


