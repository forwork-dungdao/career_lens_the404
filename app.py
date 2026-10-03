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
    page_title="TechSkill Radar - Data Visualization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_local_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.logo("🎯")
    
    # Load CSS tùy chỉnh
    load_local_css()

    # Rút ngắn và căn giữa thanh điều hướng
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col2:
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
        st.title("📊 Dashboard")
        st.write("Phân tích dữ liệu từ `job.csv`")

        # Đọc dữ liệu thực tế
        data_path = os.path.join(os.path.dirname(__file__), "data", "job.csv")
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Không thể đọc dữ liệu: {e}")
            return
            
        from itables.streamlit import interactive_table

        # Layout: 2/3 (Cột trái) - 1/3 (Cột phải)
        col_left, col_right = st.columns([2, 1])

        # ---------------------------------------------
        # Cột Trái: Biểu đồ Cột (2/3 trang)
        # ---------------------------------------------
        with col_left:
            with st.container(border=True):
                st.subheader("1. Top 10 Vị trí công việc phổ biến")
                
                # Bộ lọc Drop box (Selectbox)
                sort_bar = st.selectbox(
                    "Sắp xếp số lượng:",
                    ["Cao đến thấp", "Thấp đến cao"],
                    key="sort_bar"
                )
                
                # Xử lý dữ liệu bar
                df_bar = df['job_title'].value_counts().reset_index().head(10)
                df_bar.columns = ['Vị trí công việc', 'Số lượng']
                if sort_bar == "Thấp đến cao":
                    df_bar = df_bar.sort_values(by='Số lượng', ascending=True)
                else:
                    df_bar = df_bar.sort_values(by='Số lượng', ascending=False)
                    
                # Vẽ biểu đồ
                fig_bar = px.bar(
                    df_bar, 
                    x='Vị trí công việc', 
                    y='Số lượng', 
                    color='Số lượng',
                    height=500
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Hiển thị bảng
                with st.expander("Hiển thị dữ liệu bảng"):
                    interactive_table(df_bar, classes="display compact", maxBytes=0)

        # ---------------------------------------------
        # Cột Phải: Biểu đồ Donut & Heatmap (1/3 trang)
        # ---------------------------------------------
        with col_right:
            # Biểu đồ Donut
            with st.container(border=True):
                st.subheader("2. Phân bố Cấp độ (Level)")
                
                # Xử lý dữ liệu donut
                df_donut = df['level'].value_counts().reset_index()
                df_donut.columns = ['Cấp độ', 'Số lượng']
                
                fig_donut = px.pie(
                    df_donut, 
                    names='Cấp độ', 
                    values='Số lượng', 
                    hole=0.5,
                    height=250
                )
                fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_donut, use_container_width=True)
                
                with st.expander("Dữ liệu bảng"):
                    interactive_table(df_donut, classes="display compact", maxBytes=0)
            
            # Biểu đồ Heatmap
            with st.container(border=True):
                st.subheader("3. Mức lương trung bình")
                
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
                    labels=dict(x="Địa điểm", y="Cấp độ", color="Lương (Tr)"),
                    aspect="auto",
                    height=250
                )
                fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_heat, use_container_width=True)
                
                with st.expander("Dữ liệu bảng"):
                    interactive_table(heatmap_data.reset_index(), classes="display compact", maxBytes=0)



    def salary_predictor():
        st.title("📄 Salary Predictor")
        left2, right2 = st.columns([7, 5])
        
        with right2:
            with st.container(border=True):
                st.subheader("Tải CV của bạn")
                st.caption("Định dạng: .pdf, .docx - Giới hạn: 10 MB - Kéo thả hoặc bấm chọn")
                up = st.file_uploader("Chọn file CV", type=["pdf", "docx"], label_visibility="collapsed")
        
        def cv_reader(file):
            pass    # Placeholder for CV reading logic

        cv_ready = False
        if up is not None:
            with st.status("AI đang trích xuất dữ liệu kỹ năng và kinh nghiệm…", expanded=True) as s:
                bar = st.progress(0)
                for p in range(0, 101, 20):
                    bar.progress(p)
                    time.sleep(0.15)
                text = cv_reader(up)
                s.update(label="Trích xuất hoàn tất.", state="complete")
        elif st.session_state.get("cv_done"):
            cv_ready = True
    
        with left2:
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
                pass


    if selected == "Dashboard":
        dashboard()
    elif selected == "Salary Predictor":
        salary_predictor()

if __name__ == "__main__":
    main()

