import os
import tempfile
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
        st.title("📊 Salary Dashboard")
        st.caption("Sample data · illustrative salary and listing figures (USD per month)")

        df = pd.DataFrame([
            {"Month": "2026-01", "Role": "Data Analyst", "Salary": 4200, "Listings": 18},
            {"Month": "2026-01", "Role": "Software Engineer", "Salary": 6100, "Listings": 25},
            {"Month": "2026-01", "Role": "UX Designer", "Salary": 4800, "Listings": 12},
            {"Month": "2026-02", "Role": "Data Analyst", "Salary": 4350, "Listings": 20},
            {"Month": "2026-02", "Role": "Software Engineer", "Salary": 6300, "Listings": 28},
            {"Month": "2026-02", "Role": "UX Designer", "Salary": 4950, "Listings": 14},
            {"Month": "2026-03", "Role": "Data Analyst", "Salary": 4500, "Listings": 22},
            {"Month": "2026-03", "Role": "Software Engineer", "Salary": 6500, "Listings": 30},
            {"Month": "2026-03", "Role": "UX Designer", "Salary": 5100, "Listings": 16},
            {"Month": "2026-04", "Role": "Data Analyst", "Salary": 4650, "Listings": 24},
            {"Month": "2026-04", "Role": "Software Engineer", "Salary": 6700, "Listings": 32},
            {"Month": "2026-04", "Role": "UX Designer", "Salary": 5250, "Listings": 18},
        ])

        total_listings = df["Listings"].sum()
        average_salary = (df["Salary"] * df["Listings"]).sum() / total_listings
        highest_salary = df["Salary"].max()

        listings_card, average_card, highest_card = st.columns(3)
        listings_card.metric("Total listings", f"{total_listings:,}")
        average_card.metric("Average salary", f"${average_salary:,.0f} / month")
        highest_card.metric("Highest salary", f"${highest_salary:,.0f} / month")

        weighted_df = df.assign(WeightedSalary=df["Salary"] * df["Listings"])
        salary_by_role = weighted_df.groupby("Role", as_index=False)[
            ["WeightedSalary", "Listings"]
        ].sum()
        salary_by_role["AverageSalary"] = (
            salary_by_role["WeightedSalary"] / salary_by_role["Listings"]
        )

        salary_by_month = weighted_df.groupby("Month", as_index=False)[
            ["WeightedSalary", "Listings"]
        ].sum().sort_values("Month")
        salary_by_month["AverageSalary"] = (
            salary_by_month["WeightedSalary"] / salary_by_month["Listings"]
        )

        listings_by_role = df.groupby("Role", as_index=False)["Listings"].sum()

        bar_col, line_col = st.columns(2)
        with bar_col:
            st.subheader("Average salary by role")
            fig_bar = px.bar(
                salary_by_role,
                x="Role",
                y="AverageSalary",
                labels={"AverageSalary": "Average salary (USD/month)"},
            )
            st.plotly_chart(fig_bar, width="stretch")

        with line_col:
            st.subheader("Average salary by month")
            fig_line = px.line(
                salary_by_month,
                x="Month",
                y="AverageSalary",
                markers=True,
                labels={"AverageSalary": "Average salary (USD/month)"},
            )
            st.plotly_chart(fig_line, width="stretch")

        st.subheader("Listings by role")
        fig_donut = px.pie(
            listings_by_role,
            names="Role",
            values="Listings",
            hole=0.6,
        )
        st.plotly_chart(fig_donut, width="stretch")


    def salary_predictor():
        st.title("📄 Salary Predictor")
        left2, right2 = st.columns([7, 5])
        
        with right2:
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
            # more placeholder
            # if text.strip() == "":
            #     st.error("Không thể đọc nội dung, vui lòng tải tệp PDF tiêu chuẩn (file scan ảnh không đọc được ký tự — OCR failed).")
            # else:
            #     st.success(f"Đã phân tích: {up.name} · {up.size/1024/1024:.1f} MB")
            #     st.session_state["cv_done"] = True
            #     st.session_state["cv_name"] = up.name
            #     st.session_state["cv_size"] = up.size
            #     st.session_state["cv_text"] = text
            #     cv_ready = True
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
