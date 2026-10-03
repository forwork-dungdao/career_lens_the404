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
    # Tiêu đề chính
    st.title("🎯 TechSkill Radar - Phân Tích Kỹ Năng & Mức Lương IT")
    st.markdown("---")

    # Sidebar cho các bộ lọc
    with st.sidebar:
        st.header("⚙️ Bộ lọc dữ liệu")
        
        # Placeholder cho các bộ lọc
        selected_role = st.selectbox(
            "Chọn vị trí công việc:",
            ["Tất cả", "Data Scientist", "Data Engineer", "Machine Learning Engineer", "Software Engineer", "Backend Developer"]
        )
        
        selected_experience = st.slider(
            "Số năm kinh nghiệm:",
            min_value=0, max_value=20, value=(0, 20)
        )
        
        st.markdown("---")
        st.info("Hệ thống TechSkill Radar: Trực quan hóa dữ liệu và dự đoán mức lương IT dựa trên kỹ năng và CV.")

    # Tạo các tab chính cho việc phân tích
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan (Overview)", "🛠️ Phân tích Kỹ năng (Skills)", "💰 Phân bố Mức lương (Salary)"])

    with tab1:
        st.header("Tổng quan thị trường")
        
        # Các metrics chính (KPIs)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng số mẫu dữ liệu", "0")
        with col2:
            st.metric("Mức lương trung bình", "0 USD")
        with col3:
            st.metric("Kỹ năng phổ biến nhất", "N/A")
        with col4:
            st.metric("Vị trí phổ biến nhất", "N/A")
        
        st.markdown("---")
        
        # Container cho biểu đồ tổng quan
        st.subheader("Xu hướng số lượng công việc theo thời gian/vị trí")
        st.info("💡 Biểu đồ Line/Bar chart thể hiện xu hướng sẽ được vẽ tại đây (sử dụng Plotly).")
        
    with tab2:
        st.header("Phân tích Kỹ năng Yêu cầu")
        
        col_skill_1, col_skill_2 = st.columns(2)
        
        with col_skill_1:
            st.subheader("Top Kỹ năng Cốt lõi (Hard Skills)")
            st.info("💡 Biểu đồ Bar chart (Ngang) thể hiện top các kỹ năng được yêu cầu nhiều nhất.")
            
        with col_skill_2:
            st.subheader("Mức lương theo kỹ năng cụ thể")
            st.info("💡 Biểu đồ Box plot hoặc Bar chart thể hiện mức lương trung bình khi có kỹ năng cụ thể.")

    with tab3:
        st.header("Phân bố Mức lương")
        
        st.subheader("Phân bố lương theo vị trí công việc")
        st.info("💡 Biểu đồ Histogram hoặc Box plot thể hiện sự phân tán của mức lương.")
        
        st.markdown("---")
        st.subheader("Góc dự đoán (Inference)")
        st.write("Tại đây có thể tích hợp form để upload CV (PDF) hoặc nhập kỹ năng để test model dự đoán.")

if __name__ == "__main__":
    main()
