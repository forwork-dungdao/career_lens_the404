import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="TechSkill Radar - Data Visualization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.logo("🎯")


    def dashboard():
        st.title("📊 Dashboard")
        st.write("Placeholder")

        # Example data
        df_bar = pd.DataFrame({
            "Category": ["A", "B", "C", "D", "E"],
            "Value": [120, 180, 90, 220, 160],
        })

        df_donut = pd.DataFrame({
            "Category": ["Desktop", "Mobile", "Tablet"],
            "Value": [55, 35, 10],
        })

        # Left / Right layout
        col1, col2 = st.columns(2)

        # Bar chart - Left
        with col1:
            st.subheader("Sales by Category")

            fig_bar = px.bar(
                df_bar,
                x="Category",
                y="Value",
            )

            st.plotly_chart(
                fig_bar,
                width='stretch',
            )

        # Donut chart - Right
        with col2:
            st.subheader("Device Distribution")

            fig_donut = px.pie(
                df_donut,
                names="Category",
                values="Value",
                hole=0.6,
            )

            st.plotly_chart(
                fig_donut,
                width='stretch',
            )


    def salary_predictor():
        st.title("📄 Salary Predictor")
        left2, right2 = st.columns([7, 5])
        
        with right2:
            st.subheader("Tải CV của bạn")
            st.caption("Định dạng: .pdf, .docx - Giới hạn: 10 MB - Kéo thả hoặc bấm chọn")
            up = st.file_uploader("Chọn file CV", type=["pdf", "docx"], label_visibility="collapsed")
            if st.button("Xóa CV — quay về Locked", use_container_width=True):
                st.session_state.pop("cv_done", None)
                st.session_state.pop("cv_name", None)
                st.rerun()
        
        def cv_reader(file):
            pass    # Placeholder for CV reading logic

        cv_ready = False
        if up is not None:
            with st.status("AI đang trích xuất dữ liệu kỹ năng và kinh nghiệm…", expanded=True) as s:
                bar = st.progress(0)
                for p in range(0, 101, 20):
                    bar.progress(p)
                    time.sleep(0.15)
                text = read_cv_text(up)
                s.update(label="Trích xuất hoàn tất.", state="complete")
            if text.strip() == "":
                st.error("Không thể đọc nội dung, vui lòng tải tệp PDF tiêu chuẩn (file scan ảnh không đọc được ký tự — OCR failed).")
            else:
                st.success(f"Đã phân tích: {up.name} · {up.size/1024/1024:.1f} MB")
                st.session_state["cv_done"] = True
                st.session_state["cv_name"] = up.name
                st.session_state["cv_size"] = up.size
                st.session_state["cv_text"] = text
                cv_ready = True
        elif st.session_state.get("cv_done"):
            cv_ready = True


    pages = [
        st.Page(dashboard, title="Dashboard", icon="📊", default=True),
        st.Page(salary_predictor, title="Salary Predictor", icon="💰"),
    ]

    pg = st.navigation(
        pages,
        position="top",
    )

    pg.run()

if __name__ == "__main__":
    main()
