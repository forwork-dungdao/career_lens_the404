import re
from pathlib import Path
import joblib
import shap
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import json

def _find_root() -> Path:
    """Tim thu muc goc cua project (thu muc chua `models/` va `data/`).

    - Truong hop chuan: file nay nam o `core/SHAP_TreeExplainer.py`
      thi root la cha cua `core/`.
    - Truong hop file bi copy di cho khac: di nguoc len cac thu muc cha,
      thu muc nao chua `models/` hoac ten la `career_lens_the404` thi lay.
    """
    p = Path(__file__).resolve()
    # truong hop chuan: core/SHAP_TreeExplainer.py -> root la cha cua core/
    if p.parent.name == "core":
        return p.parent.parent
    # file dang nam o root hoac cho khac: do len tim thu muc co models/
    for anc in [p.parent, *p.parents]:
        if (anc / "models").exists():
            return anc
        if anc.name == "career_lens_the404":
            return anc
    return p.parent


# --- Duong dan dung chung cho ca file ---
ROOT = _find_root()          # vd: .../career_lens_the404/career_lens_the404
PICKLE_DIR = ROOT / "models"  # noi chua 3 file model/encoder
DATA_DIR = ROOT / "data"      # noi chua job.csv (du lieu thi truong)

# Anh xa ten logic -> ten file thuc te tren dia.
# Phai khop tuyet doi voi file trong thu muc models/,
# neu sai 1 ky tu se roi vao FileNotFoundError o ham load_pickle.
PICKLE_FILES = {
    "gbm_model": "gbm_model.pkl",
    "mlb": "mlb.pkl",
    "feature_names": "feature_names.pkl",
}


def load_pickle(stem: str):
    """Nap 1 artifact theo key trong PICKLE_FILES.

    Vi du: load_pickle("gbm_model") -> doc file models/gbm_model.pkl.
    Neu file khong ton tai thi bao ro ca duong dan + ROOT de de debug.
    """
    path = PICKLE_DIR / PICKLE_FILES[stem]
    if not path.exists():
        raise FileNotFoundError(f"Khong thay {path} | ROOT={ROOT}")
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Loi khi nap pickle tu {path}: {str(e)}")
        raise


def resolve_data() -> Path:
    """Tra ve duong dan file du lieu thi truong `data/job.csv`.

    File nay dung de lay mau thi truong (market context) khi giai thich SHAP.
    """
    path = DATA_DIR / "job.csv"
    if not path.exists():
        raise FileNotFoundError(f"Khong thay {path} | ROOT={ROOT}")
    return path


# --- Nap artifact 1 lan duy nhat khi import module ---
# gbm: mo hinh LightGBMRegressor du doan avg_salary.
# mlb: MultiLabelBinarizer, chua taxonomy ky nang o mlb.classes_.
# feature_names: thu tu 579 cot luc train, bat buoc phai giu dung thu tu khi predict.
gbm = load_pickle("gbm_model")
mlb = load_pickle("mlb")
feature_names = list(load_pickle("feature_names"))
DATA_PATH = resolve_data()


def clean_col(name) -> str:
    """Chuan hoa ten cot cho LightGBM/pandas.

    LightGBM khong ua ky tu dac biet (C#, C++, .NET, ...) nen thay
    moi cum ky tu khong phai [A-Za-z0-9_] bang dau "_" .
    Vi du: "C#" -> "C_", "ASP.NET Core" -> "ASP_NET_Core".
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


def _dedup_names(raw_cols, seen) -> list:
    """Dat lai ten trung nhau theo dung logic luc train.

    Tai sao can ham nay? Vi clean_col co the bien 2 ten khac nhau
    thanh 1 ten giong nhau (vd "C#" va "C++" deu -> "C_").
    Luc train (salary_predictor.py) gap trung thi doi ten thanh _1, _2...
    nen o day phai lam y het, neu khong DataFrame se co 2 cot trung ten
    va `reindex` se loi "cannot reindex on an axis with duplicate labels".

    - raw_cols: danh sach ten goc can chuan hoa.
    - seen: danh sach ten da duoc dung truoc do (de tranh trung cheo).
    """
    # Dedup giong logic luc train trong salary_predictor.py:
    # gap ten da thay thi them _1, _2, ...
    seen = list(seen)
    out = []
    for raw in raw_cols:
        base = clean_col(raw)
        if base in seen or base in out:
            i = 1
            while f"{base}_{i}" in seen or f"{base}_{i}" in out:
                i += 1
            base = f"{base}_{i}"
        out.append(base)
    return out


# --- Bang anh xa skill -> ten cot sau khi clean + dedup ---
# Vi du thuc te: "C#" -> "C_", "C++" -> "C__1" (khop voi feature_names trong pickle).
# Neu dung dict naive {s: clean_col(s)} thi ca 2 deu -> "C_",
# tao ra 2 cot "C_" giong nhau va gay loi reindex.
_seen_cols: list = []
skill_to_col = {}
for _s in mlb.classes_:
    _c = _dedup_names([_s], _seen_cols)[0]
    _seen_cols.append(_c)
    skill_to_col[_s] = _c
_SKILL_COLS = list(_seen_cols)  # thu tu cot skill, dung khi tao DataFrame
col_to_skill = {v: k for k, v in skill_to_col.items()}  # anh xa nguoc de hien thi


def pretty(name: str) -> str:
    """Doi ten cot da clean ve ten skill dep de hien thi.

    Vi du: "ASP_NET_Core" -> "ASP.NET Core". Neu khong tim thay
    trong bang anh xa thi giu nguyen ten goc.
    """
    return col_to_skill.get(name, name)


def encode_cv(cv_dict: dict) -> pd.DataFrame:
    """Ma hoa 1 CV thanh DataFrame 1 dong, dung thu tu `feature_names`.

    Dau vao cv_dict gom: job_title, level, location, years_experience, skills.
    Cac buoc:
    1. Skills: dung mlb.transform (one-hot theo taxonomy da train).
    2. job_title/level/location: one-hot bang pd.get_dummies.
    3. years_experience: giu nguyen so.
    4. Ghep 3 khoi lai, xoa cot trung (neu co), roi reindex ve
       dung 579 cot cua feature_names (thieu cot nao thi dien 0).
    """
    raw = cv_dict.get("skills", [])
    if isinstance(raw, str):
        raw = raw.split(",")
    skill_list = [str(s).strip() for s in raw if str(s).strip()]

    # One-hot skills theo dung thu tu _SKILL_COLS da dedup.
    skill_arr = mlb.transform([skill_list])
    skill_df = pd.DataFrame(skill_arr, columns=_SKILL_COLS)

    # One-hot 3 cot categorical cua 1 CV.
    cat_df = pd.get_dummies(pd.DataFrame([{
        "job_title": cv_dict.get("job_title"),
        "level": cv_dict.get("level"),
        "location": cv_dict.get("location"),
    }]))
    # Dedup ten cot categorical theo sau cot skill (tranh trung cheo).
    cat_df.columns = _dedup_names(list(cat_df.columns), _SKILL_COLS)

    # Khởi tạo DataFrame toàn số 0 theo đúng số lượng và thứ tự cột lúc train
    X_input = pd.DataFrame(0, index=[0], columns=feature_names)

    # Bơm dữ liệu skills (One-hot) vào đúng tọa độ
    s_cols = [c for c in skill_df.columns if c in X_input.columns]
    if s_cols:
        X_input[s_cols] = skill_df[s_cols]

    # Bơm dữ liệu categorical (Job title, Level, Location)
    cat_cols = [c for c in cat_df.columns if c in X_input.columns]
    if cat_cols:
        X_input[cat_cols] = cat_df[cat_cols]

    # Bơm dữ liệu dạng số (Years of Experience)
    if "years_experience" in X_input.columns:
        X_input.at[0, "years_experience"] = cv_dict.get("years_experience", 0)

    return X_input


# --- Cache dung chung: chi khoi tao explainer/base_value/market 1 lan ---
_explainer = None      # doi tuong shap.TreeExplainer, tao lazy
_base_value = None     # gia tri ky vong (expected_value) cua model = luong baseline
_market_cache = {}     # cache ket qua get_market_context theo (DATA_PATH, n_sample)


def get_explainer():
    """Tra ve shap.TreeExplainer dung chung (lazy init).

    TreeExplainer hieu cau truc cay cua LightGBM nen tinh SHAP nhanh
    hon KernelExplainer rat nhieu.
    """
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(gbm)
    return _explainer


def get_base_value() -> float:
    """Tra ve luong baseline (expected_value) cua model.

    Day la diem xuat phat cua moi giai thich SHAP:
    predicted_salary = base_value + tong(shap_values).
    """
    global _base_value
    if _base_value is None:
        ev = get_explainer().expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.asarray(ev).ravel()[0]
        _base_value = float(ev)
    return _base_value


def _shap_2d(explainer, X: pd.DataFrame) -> np.ndarray:
    """Tinh SHAP values va luon tra ve mang 2D (n_samples, n_features).

    Tuong thich 2 API cua shap:
    - ban cu: explainer.shap_values(X) (co the tra ve list voi multi-output).
    - ban moi: explainer(X).values (co the tra ve mang 3D).
    Ham nay chuan hoa ca 2 ve dang 2D de code phia sau khoi phan nhanh.
    """
    if hasattr(explainer, "shap_values"):
        out = explainer.shap_values(X)
    else:
        out = explainer(X).values
    arr = np.asarray(out)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def get_market_context(n_sample: int = 1000):
    """Lay boi canh thi truong: mau X, du doan, do quan trong global, top skills.

    - Doc job.csv, encode toan bo thi truong bang cung logic nhu encode_cv.
    - Lay ngau nhien toi da n_sample dong de tinh SHAP cho nhanh.
    - global_importance: trung binh |SHAP| cua moi feature (feature nao
      anh huong luong manh nhat tren toan thi truong).
    - top_global: top 10 trong so do nhung chi lay cac cot skill.
    - market_preds: luong model du doan cho tung dong mau.
    Ket qua duoc cache theo (DATA_PATH, n_sample) de goi lan 2 khong phai tinh lai.
    """
    key = (str(DATA_PATH), n_sample)
    if key in _market_cache:
        return _market_cache[key]

    df = pd.read_csv(DATA_PATH)
    skill_col = "skill" if "skill" in df.columns else "skills"
    df["skill_list"] = df[skill_col].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notnull(x) else []
    )
    # Encode skills cua toan thi truong, dung cung _SKILL_COLS da dedup.
    s_df = pd.DataFrame(
        mlb.transform(df["skill_list"]),
        columns=_SKILL_COLS,
    )
    cat_all = pd.get_dummies(df[["job_title", "level", "location"]])
    cat_all.columns = _dedup_names(list(cat_all.columns), _SKILL_COLS)
    num_all = df[["years_experience"]]
    
    # Khởi tạo DataFrame toàn số 0 cho tập thị trường
    X_all = pd.DataFrame(0, index=df.index, columns=feature_names)
    
    # Bơm dữ liệu theo từng block
    s_cols = [c for c in s_df.columns if c in X_all.columns]
    if s_cols:
        X_all[s_cols] = s_df[s_cols]
        
    cat_cols = [c for c in cat_all.columns if c in X_all.columns]
    if cat_cols:
        X_all[cat_cols] = cat_all[cat_cols]
        
    if "years_experience" in X_all.columns:
        X_all["years_experience"] = num_all["years_experience"]

    X_sample = X_all.sample(min(n_sample, len(X_all)), random_state=42)
    shap_sample = _shap_2d(get_explainer(), X_sample)
    global_importance = pd.Series(
        np.abs(shap_sample).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)
    top_global = global_importance[
        global_importance.index.isin(list(skill_to_col.values()))
    ].head(10)
    market_preds = np.asarray(gbm.predict(X_sample))

    ctx = (X_sample, market_preds, global_importance, top_global)
    _market_cache[key] = ctx
    return ctx


def explain_cv(cv_dict: dict, include_detail: bool = False):
    """Giai thich 1 CV: du doan luong + diem manh/yeu + vi tri phan tram.

    Cong thuc kiem tra tinh dung dan cua encode:
        predicted_salary gan bang base_value + tong(shap_values).
    Neu lech qua 0.01 thi encode dang sai (sai thu tu cot / sai dedup).

    Tra ve dict 2 nhom:
    - market_baseline: luong baseline + top ky nang dang gia nhat thi truong.
    - user_cv_valuation: luong du doan cua CV, phan tram so voi thi truong,
      top 5 strengths (shap > 0 va CV co feature do), top 5 weaknesses (shap < 0).
    - Neu include_detail=True thi kem them bang detail day du de debug.
    """
    x_cv = encode_cv(cv_dict)
    base_value = get_base_value()
    _, market_preds, _, top_global = get_market_context()

    shap_cv = _shap_2d(get_explainer(), x_cv)[0]
    pred = float(np.asarray(gbm.predict(x_cv)).ravel()[0])

    if abs(pred - (base_value + shap_cv.sum())) > 1e-2:
        raise ValueError(f"Lech encode: pred={pred}, base+sum={base_value + shap_cv.sum()}")

    detail = pd.DataFrame({
        "feature": feature_names,
        "value": x_cv.iloc[0].values,
        "shap": shap_cv,
    }).sort_values("shap", ascending=False)

    strengths = detail[(detail["shap"] > 0) & (detail["value"] != 0)]
    weaknesses = detail[detail["shap"] < 0].head(5)
    percentile = float(percentileofscore(market_preds, pred))

    out = {
        "market_baseline": {
            "base_salary": base_value,
            "top_global_skills": [pretty(c) for c in top_global.index.tolist()],
        },
        "user_cv_valuation": {
            "predicted_salary": pred,
            "market_percentile": round(percentile, 1),
            "strengths_added_value": [
                {"skill": pretty(r["feature"]), "impact": float(r["shap"])}
                for _, r in strengths.head(5).iterrows()
            ],
            "weaknesses_deducted_value": [
                {"skill": pretty(r["feature"]), "impact": float(r["shap"])}
                for _, r in weaknesses.iterrows()
            ],
        },
    }
    if include_detail:
        out["_debug_detail"] = detail
    return out


def plot_waterfall(cv_dict: dict, max_display: int = 12):
    """Ve bieu do SHAP waterfall cho 1 CV.

    - Dung backend "Agg" (khong can cua so GUI) nen chay duoc ca khi debug.
    - Truc waterfall di tu base_value (luong baseline), moi thanh la 1 feature
      day luong len/xuong, cuoi cung cham toi predicted_salary.
    - max_display: chi ve toi da bao nhieu feature quan trong nhat.
    """
    matplotlib.use("Agg")

    x_cv = encode_cv(cv_dict)
    shap_cv = _shap_2d(get_explainer(), x_cv)[0]
    exp = shap.Explanation(
        values=shap_cv,
        base_values=get_base_value(),
        data=x_cv.iloc[0].values,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo chay truc tiep file nay de test nhanh pipeline:
    # encode -> predict -> SHAP -> in ket qua JSON ra terminal.
    demo = {
        "job_title": "Backend Developer",
        "level": "Mid",
        "location": "Hanoi",
        "years_experience": 2.5,
        "skills": ["Python", "Django", "PostgreSQL"],
    }
    print(f"ROOT={ROOT} | DATA={DATA_PATH}")
    res = explain_cv(demo)
    print(json.dumps(
        {k: v for k, v in res.items() if not k.startswith("_")},
        indent=2, ensure_ascii=False
    ))
