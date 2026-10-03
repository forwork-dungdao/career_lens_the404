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
def load_skill_percentages():
    skill_csv_path = os.path.join(os.path.dirname(__file__), "data", "skill_listed.csv")
    if os.path.exists(skill_csv_path):
        return pd.read_csv(skill_csv_path)
    from core.skill_listed import generate_skill_percentage_df
    return generate_skill_percentage_df()

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
                    "background-color": "#ffffff", 
                    "border-radius": "16px", 
                    "border": "1px solid #e2e8f0",
                    "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05)"
                },
                "icon": {"color": "#475569", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "center", 
                    "margin": "0px", 
                    "padding": "8px 16px",
                    "--hover-color": "#f1f5f9",
                    "border-radius": "12px",
                    "font-family": "'Inter', sans-serif"
                },
                "nav-link-selected": {
                    "background-color": "#0f172a", 
                    "color": "#ffffff"
                },
            }
        )

    def dashboard():
        st.title("DASHBOARD")

        # Đọc dữ liệu thực tế
        data_path = os.path.join(os.path.dirname(__file__), "data", "job.csv")
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Không thể đọc dữ liệu: {e}")
            return

        # Layout: 2/3 (Cột trái) - 1/3 (Cột phải)
        col_left, col_right = st.columns([2, 1])

        # ---------------------------------------------
        # Cột Trái: Biểu đồ Cột Ngang Kỹ năng (2/3 trang)
        # ---------------------------------------------
        with col_left:
            with st.container(border=True):
                st.subheader("TOP NHỮNG KỸ NĂNG PHỔ BIẾN NHẤT")
                
                # Bộ lọc Drop box (Selectbox) sắp xếp
                sort_bar = st.selectbox(
                    "Sắp xếp theo tỷ lệ:",
                    ["Cao đến thấp", "Thấp đến cao"],
                    key="sort_skill"
                )
                
                # Đọc dữ liệu kỹ năng từ skill_listed
                df_skill = load_skill_percentages()
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
                    height=500
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
                st.subheader("PHÂN BỐ CẤP ĐỘ")
                
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
                st.subheader("MỨC LƯƠNG TRUNG BÌNH")
                
                # Drop box lọc số lượng địa điểm
                top_n_loc = st.selectbox("Số lượng địa điểm:", [3, 5, 7], index=1)
                
                # Lấy top N locations
                top_locs = df['location'].value_counts().head(top_n_loc).index
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
                    labels=dict(x="Địa điểm", y="Cấp độ", color="Lương (Triệu đồng)"),
                    aspect="auto",
                    height=220
                )
                fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_heat, width='stretch')



    def salary_predictor():
        st.title("SALARY PREDICTOR")
        result, upload = st.columns([7, 5])
        
        with upload:
            with st.container(border=True):
                st.subheader("Tải CV của bạn")
                st.caption("Định dạng: .pdf, .docx - Giới hạn: 10 MB - Kéo thả hoặc bấm chọn")
                up = st.file_uploader("Chọn file CV", type=["pdf", "docx"], label_visibility="collapsed")
        
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

        cv_ready = False
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
            cv_ready = True
        elif st.session_state.get("cv_done"):
            cv_ready = True
    
        with result:
            if not cv_ready and not st.session_state.get("cv_done"):
                # before cv upload
                st.markdown(
                    """
                    <div style="border:2px dashed #cbd5e1;background:#f8fafc;border-radius:16px;min-height:420px;
                                display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px 24px;color:#64748b">
                    <div style="width:74px;height:74px;border-radius:50%;background:#e2e8f0;display:flex;align-items:center;
                                justify-content:center;font-size:34px;margin-bottom:14px">🔒</div>
                    <h3 style="color:#0f172a;margin:0 0 8px">Biểu đồ đang khóa</h3>
                    <p style="max-width:420px">Vui lòng tải lên CV của bạn ở bên phải để hệ thống AI phân tích năng lực và kích hoạt biểu đồ dự báo.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                cv_data = st.session_state.get("cv_data")
                if cv_data:
                    import joblib
                    import re
                    
                    try:
                        model_dir = os.path.join(os.path.dirname(__file__), "models")
                        mlb = joblib.load(os.path.join(model_dir, "mlb.pkl"))
                        gbm_model = joblib.load(os.path.join(model_dir, "gbm_model.pkl"))
                        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
                        
                        # Khởi tạo DataFrame toàn số 0 theo đúng chuẩn
                        X_input = pd.DataFrame(0, index=[0], columns=feature_names)
                        
                        # 1. Điền thông tin kỹ năng
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
                        
                        # 2. Điền số năm kinh nghiệm
                        years = float(cv_data.get("năm kinh nghiệm", 0.0))
                        if "years_experience" in X_input.columns:
                            X_input.at[0, "years_experience"] = years
                            
                        # 3. Điền cấp độ (Level)
                        level = cv_data.get("level công việc", "")
                        if level == "Mid-Level" and "level_Middle" in X_input.columns:
                            X_input.at[0, "level_Middle"] = 1
                        elif level == "Junior" and "level_Junior" in X_input.columns:
                            X_input.at[0, "level_Junior"] = 1
                        elif level == "Senior" and "level_Senior" in X_input.columns:
                            X_input.at[0, "level_Senior"] = 1
                            
                        # Dự đoán
                        pred_salary = gbm_model.predict(X_input)[0]
                        
                        st.markdown("### Kết quả Phân tích CV & Dự đoán Lương")
                        st.markdown(
                            f'''
                            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 24px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
                                <h4 style="color: #94a3b8; margin: 0 0 12px 0; font-weight: 500;">MỨC LƯƠNG ĐỀ XUẤT (VNĐ)</h4>
                                <h1 style="color: #10b981; margin: 0; font-size: 48px; font-weight: 700;">{pred_salary:,.1f} Triệu</h1>
                                <p style="color: #cbd5e1; margin: 12px 0 0 0;">Dựa trên phân tích bằng Machine Learning</p>
                            </div>
                            ''', 
                            unsafe_allow_html=True
                        )
                        
                        col_e, col_l = st.columns(2)
                        with col_e:
                            st.metric(label="Kinh nghiệm", value=f"{years} năm")
                        with col_l:
                            st.metric(label="Cấp độ", value=level)
                            
                        st.markdown("**Kỹ năng phát hiện được:**")
                        if skills:
                            skills_html = "".join([f'<span style="display:inline-block; background:#e2e8f0; color:#0f172a; padding:4px 12px; border-radius:16px; margin:4px; font-size:14px; font-weight:500;">{s}</span>' for s in skills])
                            st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
                        else:
                            st.info("Không tìm thấy kỹ năng IT cụ thể.")
                            
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")


    if selected == "Dashboard":
        dashboard()
    elif selected == "Salary Predictor":
        salary_predictor()

if __name__ == "__main__":
    main()
