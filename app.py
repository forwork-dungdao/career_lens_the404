import os
import tempfile
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Thêm các thư viện UI đã cài đặt
from streamlit_option_menu import option_menu
import streamlit_antd_components as sac
import streamlit_shadcn_ui as ui
import extra_streamlit_components as stx
from st_aggrid import AgGrid
import itables
import pygwalker as pyg

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Career Lens",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_local_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_data
def get_skills_by_job_title(df: pd.DataFrame, job_title: str) -> pd.DataFrame:
    from collections import Counter
    if job_title != "Tất cả các ngành":
        df = df[df['job_title'] == job_title]
    
    all_skills = []
    for skill_str in df['skills'].dropna():
        skills = [s.strip() for s in skill_str.split(',')]
        all_skills.extend(skills)
        
    counts = Counter(all_skills)
    total_jobs = len(df)
    
    if total_jobs == 0 or not counts:
        return pd.DataFrame(columns=['Skill', 'Count', 'Percentage (%)'])
        
    skill_df = pd.DataFrame({
        'Skill': list(counts.keys()),
        'Count': list(counts.values()),
    })
    skill_df['Percentage (%)'] = (skill_df['Count'] / total_jobs) * 100
    return skill_df.sort_values(by='Percentage (%)', ascending=False).reset_index(drop=True)

def main():
    st.logo("🎯")
    
    # Load CSS tùy chỉnh
    load_local_css()

    # Rút ngắn và căn giữa thanh điều hướng
    left, middle, right = st.columns([1, 2, 1])
    with middle:
        selected = option_menu(
            menu_title=None, 
            options=["Dashboard", "Salary Predictor"],
            icons=["bar-chart-line", "cash-coin"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "4px!important", 
                    "background-color": "transparent", 
                    "border-radius": "16px", 
                    "border": "1px solid var(--secondary-background-color)"
                },
                "icon": {"color": "var(--text-color)", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "center", 
                    "margin": "0px", 
                    "padding": "8px 16px",
                    "--hover-color": "var(--secondary-background-color)",
                    "border-radius": "12px",
                    "font-family": "'Inter', sans-serif",
                    "color": "var(--text-color)"
                },
                "nav-link-selected": {
                    "background-color": "var(--primary-color)", 
                    "color": "#ffffff"
                },
            }
        )

    def dashboard():
        st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>DASHBOARD</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.6; margin-bottom: 2rem;'>Phân tích tổng quan thị trường tuyển dụng IT</p>", unsafe_allow_html=True)

        # Đọc dữ liệu thực tế
        data_path = os.path.join(os.path.dirname(__file__), "data", "job.csv")
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Không thể đọc dữ liệu: {e}")
            return

        # ── KPI METRICS ROW ──
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            with st.container(border=True):
                st.markdown(f"""<div class="kpi-card">
                    <h2>📊 {len(df):,}</h2>
                    <p>Tổng tin tuyển dụng</p>
                </div>""", unsafe_allow_html=True)
        with kpi2:
            with st.container(border=True):
                st.markdown(f"""<div class="kpi-card">
                    <h2>💼 {df['job_title'].nunique()}</h2>
                    <p>Ngành nghề</p>
                </div>""", unsafe_allow_html=True)
        with kpi3:
            with st.container(border=True):
                avg_sal = df['avg_salary'].mean()
                st.markdown(f"""<div class="kpi-card">
                    <h2>💰 {avg_sal:.1f}M</h2>
                    <p>Lương trung bình</p>
                </div>""", unsafe_allow_html=True)
        with kpi4:
            with st.container(border=True):
                st.markdown(f"""<div class="kpi-card">
                    <h2>📍 {df['location'].nunique()}</h2>
                    <p>Khu vực</p>
                </div>""", unsafe_allow_html=True)

        # ── BỘ LỌC TỔNG QUÁT ──
        with st.container(border=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                top_jobs = df['job_title'].value_counts().head(20).index.tolist()
                job_titles = ["Tất cả các ngành"] + sorted(top_jobs)
                selected_job = st.selectbox("🏢 Ngành nghề:", job_titles, key="global_job_filter")
            with f_col2:
                max_exp = int(df['years_experience'].max()) if pd.notna(df['years_experience'].max()) else 15
                exp_options = list(range(1, max_exp + 1))
                selected_exp = st.selectbox("📅 Kinh nghiệm tối đa (năm):", exp_options, index=len(exp_options) - 1, key="global_exp_filter")
            with f_col3:
                top_n_skills = st.selectbox("📊 Số kỹ năng hiển thị:", [5, 10, 15, 20], index=1, key="top_n_skills")
                
        # Áp dụng bộ lọc cho DataFrame dùng chung
        if selected_job != "Tất cả các ngành":
            df = df[df['job_title'] == selected_job]
        df = df[df['years_experience'] <= selected_exp]
        
        if df.empty:
            st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại. Hãy điều chỉnh lại.")
            return

        # Hiển thị số lượng kết quả sau lọc
        st.caption(f"📌 Đang hiển thị **{len(df):,}** tin tuyển dụng phù hợp")

        # ── CHARTS LAYOUT ──
        col_left, col_right = st.columns([2, 1])

        # Cột Trái: Biểu đồ Cột Ngang Kỹ năng
        with col_left:
            with st.container(border=True):
                title_col, sort_col = st.columns([3, 1])
                with title_col:
                    st.markdown("<h3 style='margin: 0;'>🏆 TOP KỸ NĂNG PHỔ BIẾN</h3>", unsafe_allow_html=True)
                with sort_col:
                    sort_bar = st.selectbox(
                        "Sắp xếp:",
                        ["Cao → Thấp", "Thấp → Cao"],
                        key="sort_skill",
                        label_visibility="collapsed"
                    )
                
                # Tính toán dữ liệu kỹ năng dựa trên DataFrame đã lọc
                df_skill = get_skills_by_job_title(df, "Tất cả các ngành")
                
                if df_skill.empty:
                    st.info("Không có dữ liệu kỹ năng cho ngành này.")
                else:
                    df_top_skills = df_skill.head(top_n_skills).copy()
                    
                    ascending = sort_bar == "Thấp → Cao"
                    df_top_skills = df_top_skills.sort_values(by="Percentage (%)", ascending=ascending)
                        
                    # Vẽ biểu đồ cột ngang với animation
                    fig_bar = px.bar(
                        df_top_skills, 
                        x="Percentage (%)", 
                        y="Skill", 
                        orientation="h",
                        color="Percentage (%)",
                        color_continuous_scale="Blues",
                        text=df_top_skills["Percentage (%)"].apply(lambda v: f"{v:.1f}%"),
                        height=580,
                        hover_data={"Count": True, "Percentage (%)": ":.1f"}
                    )
                    fig_bar.update_layout(
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="Tỷ lệ xuất hiện (%)",
                        yaxis_title="",
                        margin=dict(t=10, b=20, l=10, r=20),
                        coloraxis_showscale=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=13),
                    )
                    fig_bar.update_traces(
                        textposition="outside",
                        marker_line_width=0,
                        hovertemplate="<b>%{y}</b><br>Tỷ lệ: %{x:.1f}%<extra></extra>"
                    )
                    st.plotly_chart(fig_bar, width='stretch', config={"displayModeBar": False})

        # Cột Phải: Donut & Heatmap
        with col_right:
            # Biểu đồ Donut
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin: 0;'>📊 PHÂN BỐ CẤP ĐỘ</h3>", unsafe_allow_html=True)
                
                df_donut = df['level'].value_counts().reset_index()
                df_donut.columns = ['Cấp độ', 'Số lượng']
                
                colors = ['#3b82f6', '#60a5fa', '#ef4444']
                fig_donut = px.pie(
                    df_donut, 
                    names='Cấp độ', 
                    values='Số lượng', 
                    hole=0.55,
                    height=240,
                    color_discrete_sequence=colors
                )
                fig_donut.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                fig_donut.update_traces(
                    textposition='outside',
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Số lượng: %{value}<br>Tỷ lệ: %{percent}<extra></extra>"
                )
                st.plotly_chart(fig_donut, width='stretch', config={"displayModeBar": False})
            
            # Biểu đồ Heatmap
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin: 0;'>🗺️ MỨC LƯƠNG TRUNG BÌNH</h3>", unsafe_allow_html=True)
                
                top_locs = df['location'].value_counts().head(5).index
                df_heat_filter = df[df['location'].isin(top_locs)]
                
                heatmap_data = df_heat_filter.pivot_table(
                    index='level', 
                    columns='location', 
                    values='avg_salary', 
                    aggfunc='mean'
                ).fillna(0).round(1)
                
                fig_heat = px.imshow(
                    heatmap_data, 
                    labels=dict(x="Địa điểm", y="Cấp độ", color="Lương (Tr)"),
                    aspect="auto",
                    height=315,
                    color_continuous_scale="Blues"
                )
                fig_heat.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    coloraxis_colorbar=dict(
                        title="Lương", 
                        nticks=5,
                        thickness=12
                    )
                )
                fig_heat.update_traces(
                    hovertemplate="<b>%{y} - %{x}</b><br>Lương TB: %{z:.1f} triệu<extra></extra>"
                )
                st.plotly_chart(fig_heat, width='stretch', config={"displayModeBar": False})


    def salary_predictor():
        if "cv_done" not in st.session_state:
            st.session_state["cv_done"] = False
            
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>AI SALARY PREDICTOR</h1>", unsafe_allow_html=True)
        
        def cv_reader(file):
            import tempfile
            from core.cv_reader import CVParserPipeline
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if file.name.endswith(".pdf") else ".docx") as tmp:
                tmp.write(file.getvalue())
                PDF_PATH = tmp.name
            try:
                pipeline = CVParserPipeline()
                df_cv = pipeline.run(PDF_PATH)
            finally:
                if os.path.exists(PDF_PATH):
                    os.remove(PDF_PATH)
            if df_cv.empty: return None
            return df_cv.iloc[0].to_dict()

        if not st.session_state.get("cv_done"):
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with st.container(border=True):
                    st.markdown("<h3 style='text-align: center;'>Tải lên CV của bạn</h3>", unsafe_allow_html=True)
                    st.caption("<div style='text-align: center;'>Định dạng hỗ trợ: .pdf, .docx - Giới hạn: 10 MB</div>", unsafe_allow_html=True)
                    up = st.file_uploader("Chọn file CV", type=["pdf", "docx"], label_visibility="collapsed")
                    
                    if up is not None:
                        with st.status("AI đang trích xuất dữ liệu kỹ năng và kinh nghiệm…", expanded=True) as s:
                            bar = st.progress(0)
                            for p in range(0, 101, 20):
                                bar.progress(p)
                                time.sleep(0.15)
                            cv_data = cv_reader(up)
                            st.session_state["cv_data"] = cv_data
                            st.session_state["cv_done"] = True
                            s.update(label="Trích xuất hoàn tất.", state="complete")
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            cv_data = st.session_state.get("cv_data")
            if cv_data:
                import joblib
                import re
                
                try:
                    model_dir = os.path.join(os.path.dirname(__file__), "models")
                    mlb = joblib.load(os.path.join(model_dir, "mlb.pkl"))
                    gbm_model = joblib.load(os.path.join(model_dir, "gbm_model.pkl"))
                    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
                    
                    X_input = pd.DataFrame(0, index=[0], columns=feature_names)
                    
                    skills = cv_data.get("skill", [])
                    if skills:
                        skills_encoded = mlb.transform([skills])[0]
                        for idx, class_name in enumerate(mlb.classes_):
                            if skills_encoded[idx] == 1:
                                col1 = class_name
                                col2 = class_name.replace(" ", "_").replace(".", "_")
                                col3 = re.sub(r"\W+", "_", class_name.strip())
                                
                                if col1 in X_input.columns:
                                    X_input.at[0, col1] = 1
                                elif col2 in X_input.columns:
                                    X_input.at[0, col2] = 1
                                elif col3 in X_input.columns:
                                    X_input.at[0, col3] = 1
                    
                    years = float(cv_data.get("năm kinh nghiệm", 0.0))
                    if "years_experience" in X_input.columns:
                        X_input.at[0, "years_experience"] = years
                        
                    level = cv_data.get("level công việc", "")
                    if level == "Mid-Level" and "level_Middle" in X_input.columns:
                        X_input.at[0, "level_Middle"] = 1
                    elif level == "Junior" and "level_Junior" in X_input.columns:
                        X_input.at[0, "level_Junior"] = 1
                    elif level == "Senior" and "level_Senior" in X_input.columns:
                        X_input.at[0, "level_Senior"] = 1
                        
                    pred_salary = gbm_model.predict(X_input)[0]
                    
                    st.markdown(
                        f'''
                        <div style="background: linear-gradient(135deg, var(--primary-color), var(--secondary-background-color)); border-radius: 16px; padding: 32px; color: var(--text-color); text-align: center; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);">
                            <h4 style="margin: 0 0 12px 0; font-weight: 500; letter-spacing: 1px;">MỨC LƯƠNG ĐỀ XUẤT (VNĐ)</h4>
                            <h1 style="color: #34d399; margin: 0; font-size: 56px; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{pred_salary:,.1f} Triệu</h1>
                            <p style="margin: 16px 0 0 0; font-size: 14px; opacity: 0.8;">Dựa trên phân tích bằng AI LightGBM</p>
                        </div>
                        ''', 
                        unsafe_allow_html=True
                    )
                    
                    info_col, skill_col = st.columns([1, 2])
                    with info_col:
                        with st.container(border=True):
                            st.markdown("<h3 style='text-align: center;'>Thông tin ứng viên</h3>", unsafe_allow_html=True)
                            st.metric(label="Kinh nghiệm", value=f"{years} năm")
                            st.metric(label="Cấp độ", value=level if level else "Chưa xác định")
                    
                    with skill_col:
                        with st.container(border=True):
                            st.markdown("<h3 style='text-align: center;'>Kỹ năng phát hiện được</h3>", unsafe_allow_html=True)
                            if skills:
                                skills_html = "".join([f'<span style="display:inline-block; background:var(--secondary-background-color); color:var(--text-color); border:1px solid var(--primary-color); padding:6px 14px; border-radius:20px; margin:4px; font-size:14px; font-weight:500; transition: all 0.2s;">{s}</span>' for s in skills])
                                st.markdown(f"<div style='margin-top: 10px;'>{skills_html}</div>", unsafe_allow_html=True)
                            else:
                                st.info("Không tìm thấy kỹ năng IT cụ thể.")
                                
                    if st.button("Phân tích CV khác", width="stretch", type="primary"):
                        st.session_state["cv_done"] = False
                        st.session_state["cv_data"] = None
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


    if selected == "Dashboard":
        dashboard()
    elif selected == "Salary Predictor":
        salary_predictor()

if __name__ == "__main__":
    main()
