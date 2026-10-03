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
print(df_new.head())

