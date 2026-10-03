import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
            up = st.file_uploader("Chọn file CV", type=["pdf", "docx"], label_visibility="collapsed", max_upload_size=10)
            if st.button("Xóa CV — quay về Locked", use_container_width=True):
                st.session_state.pop("cv_done", None)
                st.session_state.pop("cv_name", None)
                st.rerun()


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
