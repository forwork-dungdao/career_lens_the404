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
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>DASHBOARD</h1>", unsafe_allow_html=True)

        # Đọc dữ liệu thực tế
        data_path = os.path.join(os.path.dirname(__file__), "data", "job.csv")
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Không thể đọc dữ liệu: {e}")
            return

        # BỘ LỌC TỔNG QUÁT TƯƠNG TÁC
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; color: var(--primary-color); margin-bottom: 1rem;'>BỘ LỌC DỮ LIỆU TỔNG QUÁT</h4>", unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                top_jobs = df['job_title'].value_counts().head(20).index.tolist()
                job_titles = ["Tất cả các ngành"] + sorted(top_jobs)
                selected_job = st.selectbox("Ngành nghề (Job Title):", job_titles, key="global_job_filter")
            with f_col2:
                max_exp = int(df['years_experience'].max()) if pd.notna(df['years_experience'].max()) else 15
                selected_exp = st.slider("Số năm kinh nghiệm tối đa:", min_value=0, max_value=max_exp, value=max_exp, step=1, key="global_exp_filter")
                
        # Áp dụng bộ lọc cho DataFrame dùng chung
        if selected_job != "Tất cả các ngành":
            df = df[df['job_title'] == selected_job]
        df = df[df['years_experience'] <= selected_exp]
        
        if df.empty:
            st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại. Hãy điều chỉnh lại.")
            return

        # Layout: 2/3 (Cột trái) - 1/3 (Cột phải)
        col_left, col_right = st.columns([2, 1])

        # ---------------------------------------------
        # Cột Trái: Biểu đồ Cột Ngang Kỹ năng (2/3 trang)
        # ---------------------------------------------
        with col_left:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center;'>TOP NHỮNG KỸ NĂNG PHỔ BIẾN NHẤT</h3>", unsafe_allow_html=True)
                
                # Bộ lọc sắp xếp
                sort_bar = st.selectbox(
                    "Sắp xếp theo tỷ lệ:",
                    ["Cao đến thấp", "Thấp đến cao"],
                    key="sort_skill"
                )
                
                # Tính toán dữ liệu kỹ năng dựa trên DataFrame đã lọc
                df_skill = get_skills_by_job_title(df, "Tất cả các ngành")
                
                if df_skill.empty:
                    st.info("Không có dữ liệu kỹ năng cho ngành này.")
                else:
                    df_top_skills = df_skill.head(10).copy()
                    
                    if sort_bar == "Thấp đến cao":
                        df_top_skills = df_top_skills.sort_values(by="Percentage (%)", ascending=True)
                    else:
                        df_top_skills = df_top_skills.sort_values(by="Percentage (%)", ascending=False)
                        
                    # Vẽ biểu đồ cột ngang
                    fig_bar = px.bar(
                        df_top_skills, 
                        x="Percentage (%)", 
                        y="Skill", 
                        orientation="h",
                        color="Percentage (%)",
                        color_continuous_scale="Blues",
                        text=df_top_skills["Percentage (%)"].apply(lambda v: f"{v:.1f}%"),
                        height=580
                    )
                    fig_bar.update_layout(
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="Tỷ lệ xuất hiện (%)",
                        yaxis_title="Kỹ năng",
                        margin=dict(t=20, b=20, l=10, r=20),
                        coloraxis_showscale=False
                    )
                    fig_bar.update_traces(textposition="outside")
                    st.plotly_chart(fig_bar, width='stretch')

        # ---------------------------------------------
        # Cột Phải: Biểu đồ Donut & Heatmap (1/3 trang)
        # ---------------------------------------------
        with col_right:
            # Biểu đồ Donut
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center;'>PHÂN BỐ CẤP ĐỘ</h3>", unsafe_allow_html=True)
                
                # Xử lý dữ liệu donut
                df_donut = df['level'].value_counts().reset_index()
                df_donut.columns = ['Cấp độ', 'Số lượng']
                
                fig_donut = px.pie(
                    df_donut, 
                    names='Cấp độ', 
                    values='Số lượng', 
                    hole=0.5,
                    height=240
                )
                fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_donut, width='stretch')
            
            # Biểu đồ Heatmap
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center;'>MỨC LƯƠNG TRUNG BÌNH</h3>", unsafe_allow_html=True)
                
                # Lấy top 5 locations phổ biến nhất
                top_locs = df['location'].value_counts().head(5).index
                df_heat_filter = df[df['location'].isin(top_locs)]
                
                # Pivot table
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
                    height=315
                )
                fig_heat.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    coloraxis_colorbar=dict(
                        title="Lương", 
                        nticks=5
                    )
                )
                st.plotly_chart(fig_heat, width='stretch')



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
                    up = st.file_uploader("", type=["pdf", "docx"], label_visibility="collapsed")
                    
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
                                
                    if st.button("Phân tích CV khác", use_container_width=True, type="primary"):
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
