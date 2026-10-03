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

    def salary_predictor():
        st.title("📄 Salary Predictor")
        st.write("Placeholder")

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
