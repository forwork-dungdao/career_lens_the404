"""SHAP TreeExplainer cho CareerLens. Clone ve chay ngay, khong sua path tay."""
import re
from pathlib import Path
import joblib
import shap
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import json

def _find_root() -> Path:
    p = Path(__file__).resolve()
    # truong hop chuan: core/SHAP_TreeExplainer.py -> root la cha cua core/
    if p.parent.name == "core":
        return p.parent.parent
    # file dang nam o root hoac cho khac: do len tim thu muc co pickle/
    for anc in [p.parent, *p.parents]:
        if (anc / "pickle").exists():
            return anc
        if anc.name == "career_lens_the404":
            return anc
    return p.parent


ROOT = _find_root()
PICKLE_DIR = ROOT / "pickle"
DATA_DIR = ROOT / "data"

PICKLE_FILES = {
    "gbm_model": "gbm_model_.pkl",
    "mlb": "mlb_.pkl",
    "feature_names": "feature_names_.pkl",
}


def load_pickle(stem: str):
    path = PICKLE_DIR / PICKLE_FILES[stem]
    if not path.exists():
        raise FileNotFoundError(f"Khong thay {path} | ROOT={ROOT}")
    return joblib.load(path)


def resolve_data() -> Path:
    path = DATA_DIR / "job.csv"
    if not path.exists():
        raise FileNotFoundError(f"Khong thay {path} | ROOT={ROOT}")
    return path


gbm = load_pickle("gbm_model")
mlb = load_pickle("mlb")
feature_names = list(load_pickle("feature_names"))
DATA_PATH = resolve_data()


def clean_col(name) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


skill_to_col = {s: clean_col(s) for s in mlb.classes_}
col_to_skill = {v: k for k, v in skill_to_col.items()}


def pretty(name: str) -> str:
    return col_to_skill.get(name, name)


def encode_cv(cv_dict: dict) -> pd.DataFrame:
    raw = cv_dict.get("skills", [])
    if isinstance(raw, str):
        raw = raw.split(",")
    skill_list = [str(s).strip() for s in raw if str(s).strip()]

    skill_arr = mlb.transform([skill_list])
    skill_df = pd.DataFrame(
        skill_arr, columns=[skill_to_col[s] for s in mlb.classes_]
    )

    cat_df = pd.get_dummies(pd.DataFrame([{
        "job_title": cv_dict.get("job_title"),
        "level": cv_dict.get("level"),
        "location": cv_dict.get("location"),
    }]))
    cat_df.columns = [clean_col(c) for c in cat_df.columns]

    num_df = pd.DataFrame([{"years_experience": cv_dict.get("years_experience", 0)}])

    X_one = pd.concat([skill_df, cat_df, num_df], axis=1)
    return X_one.reindex(columns=feature_names, fill_value=0)


_explainer = None
_base_value = None
_market_cache = {}


def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(gbm)
    return _explainer


def get_base_value() -> float:
    global _base_value
    if _base_value is None:
        ev = get_explainer().expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.asarray(ev).ravel()[0]
        _base_value = float(ev)
    return _base_value


def _shap_2d(explainer, X: pd.DataFrame) -> np.ndarray:
    if hasattr(explainer, "shap_values"):
        out = explainer.shap_values(X)
    else:
        out = explainer(X).values
    arr = np.asarray(out)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def get_market_context(n_sample: int = 1000):
    key = (str(DATA_PATH), n_sample)
    if key in _market_cache:
        return _market_cache[key]

    df = pd.read_csv(DATA_PATH)
    skill_col = "skill" if "skill" in df.columns else "skills"
    df["skill_list"] = df[skill_col].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notnull(x) else []
    )
    s_df = pd.DataFrame(
        mlb.transform(df["skill_list"]),
        columns=[skill_to_col[s] for s in mlb.classes_],
    )
    cat_all = pd.get_dummies(df[["job_title", "level", "location"]])
    cat_all.columns = [clean_col(c) for c in cat_all.columns]
    num_all = df[["years_experience"]]
    X_all = pd.concat([s_df, cat_all, num_all], axis=1).reindex(
        columns=feature_names, fill_value=0
    )

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
