# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import train_and_evaluate, save_model, load_model, predict_df
from generate_sample_data import generate
import plotly.express as px
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

load_dotenv()
st.set_page_config(page_title="𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄", layout="wide")

# ---------------- Banner Section ----------------
banner_path = "aii.jpg"   # Replace with your banner image filename
if os.path.exists(banner_path):
    from PIL import Image
    banner_img = Image.open(banner_path)
    st.image(banner_img, use_container_width=True)


# ---------------- Styled Title & Subheader ----------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1; font-size: 48px;'>
𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄 – ᴡᴀᴛᴄʜ. ᴅᴇᴛᴇᴄᴛ. ᴘʀᴏᴛᴇᴄᴛ
</h1>
<p style='text-align: center; color: #555; font-size: 20px;'>Zero Dropouts, Infinite Potential</p>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown("## 📊 Data / Model Selection")
data_option = st.sidebar.radio("Choose data", ("Use sample data", "Upload Single CSV", "Upload multiple CSVs (Attendance, Tests, Fees)"))
model_option = st.sidebar.selectbox("Model action", ("Train new model", "Load existing model (dropout_model.joblib)"))

# ---------------- Data Handling ----------------
df = None

# Option 1: Sample Data
if data_option == "Use sample data":
    df = generate(2000)
    st.sidebar.success("✅ Generated sample dataset (2000 rows)")

# Option 2: Single CSV
elif data_option == "Upload Single CSV":
    uploaded_file = st.sidebar.file_uploader("Upload students CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            df = generate(2000)
    else:
        st.sidebar.info("No file uploaded. Using sample data.")
        df = generate(2000)

# Option 3: Multiple CSVs
elif data_option == "Upload multiple CSVs (Attendance, Tests, Fees)":
    st.sidebar.info("Upload 3 CSVs with a common `student_id` column")
    att_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"], key="att")
    test_file = st.sidebar.file_uploader("Upload Tests CSV", type=["csv"], key="test")
    fee_file = st.sidebar.file_uploader("Upload Fees CSV", type=["csv"], key="fee")

    if att_file and test_file and fee_file:
        try:
            df_att = pd.read_csv(att_file)[["student_id", "attendance_pct"]]
            df_test = pd.read_csv(test_file)[["student_id", "avg_test_pct", "avg_assignment_pct"]]
            df_fee = pd.read_csv(fee_file)[["student_id", "fee_delay_days"]]

            df = df_att.merge(df_test, on="student_id", how="outer")
            df = df.merge(df_fee, on="student_id", how="outer")
            st.sidebar.success(f"✅ Merged dataset with {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error merging CSVs: {e}")
            df = generate(2000)
    else:
        st.sidebar.warning("⚠ Please upload all 3 files")
        df = generate(2000)

# Required columns check & type enforcement
required_cols = ["student_id","gender","scholarship","attendance_pct","avg_assignment_pct","avg_test_pct","fee_delay_days","num_attempts","prior_arrears","engagement_score","dropout_risk"]
expected_types = {"student_id": str, "gender": str, "scholarship": int, "attendance_pct": float, "avg_assignment_pct": float, "avg_test_pct": float,
                  "fee_delay_days": int, "num_attempts": int, "prior_arrears": int, "engagement_score": float}

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Missing columns: {missing}. Using sample data may fill all required columns.")

for col, dtype in expected_types.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

st.write("### STUDENTS DATASET (up to 1000 rows)")
st.dataframe(df.head(1000))

# ---------------- Model Handling ----------------
model = None
metrics = None

if model_option == "Train new model":
    st.subheader("Train model")
    if st.button("Train model on selected dataset"):
        model, metrics = train_and_evaluate(df)
        save_model(model, "dropout_model.joblib")
        st.success("✅ Model trained and saved to dropout_model.joblib")
        with st.expander("📊 View detailed model metrics"):
            st.json(metrics)

else:
    model = load_model("dropout_model.joblib")
    st.success("✅ Loaded model dropout_model.joblib")

# ---------------- Prediction & Risk ----------------
high_risk = pd.DataFrame()  # initialize

if model is not None:
    st.subheader("Predict & Explore")
    predict_option = st.radio("Prediction source", ("Predict on dataset shown above", "Upload new students to predict"))

    if predict_option == "Predict on dataset shown above":
        pred_df = predict_df(model, df)
    else:
        new_file = st.file_uploader("Upload new students CSV", type=["csv"], key="predupload")
        if new_file:
            new_df = pd.read_csv(new_file)
            pred_df = predict_df(model, new_df)
        else:
            st.info("Upload new CSV; predictions will run on current dataset.")
            pred_df = predict_df(model, df)

    # Risk level
    def assign_risk_level(p):
        if p < 0.3: return "Low"
        elif p < 0.6: return "Medium"
        else: return "High"
    pred_df["risk_level"] = pred_df["risk_proba"].apply(assign_risk_level)

    # Color function
    def color_risk(val):
        colors = {"High":"#ff4d4d","Medium":"#ffc107","Low":"#28a745"}
        return f"background-color: {colors.get(val,'white')}; color:white; font-weight:bold;"

    # Metrics cards
    c1,c2,c3 = st.columns(3)
    c1.metric("High-Risk Students", len(pred_df[pred_df["risk_level"]=="High"]))
    c2.metric("Average Risk Probability", round(pred_df["risk_proba"].mean(),2))
    c3.metric("Low-Risk Students", len(pred_df[pred_df["risk_level"]=="Low"]))

    # Top 10 table
    st.write("### Predictions (Top 10)")
    styled = pred_df.sort_values("risk_proba", ascending=False).head(10).style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True)

    # Histogram & Pie
    st.write("### Risk Distribution")
    fig = px.histogram(pred_df, x="risk_proba", nbins=30, color="risk_level",
                       color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                       title="Predicted risk probability")
    st.plotly_chart(fig, use_container_width=True)

    pie_fig = px.pie(pred_df, names="risk_level", color="risk_level",
                     color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                     title="Risk Level Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

    # High-risk table with threshold
    if "threshold" not in st.session_state: st.session_state.threshold=0.5
    threshold = st.slider("Risk threshold",0.0,1.0,st.session_state.threshold)
    st.session_state.threshold=threshold
    high_risk = pred_df[pred_df["risk_proba"]>=threshold].sort_values("risk_proba",ascending=False)
    st.write(f"High-risk students (risk_proba>={threshold}): {len(high_risk)}")
    st.dataframe(high_risk[["student_id","attendance_pct","avg_test_pct","engagement_score","risk_proba","risk_level"]].head(50).style.applymap(color_risk, subset=["risk_level"]))

    # Download buttons
    st.download_button("📥 Download High-risk CSV", high_risk.to_csv(index=False), file_name="high_risk_students.csv", mime="text/csv")
    st.download_button("📥 Download All Predictions", pred_df.to_csv(index=False), file_name="all_predictions.csv", mime="text/csv")

   # ---------------- Counseling / Outreach Email Section (Async + Progress Bar) ----------------
st.write("## Counseling / Outreach ")
with st.expander("Compose Email"):
    enable_email = st.checkbox("Enable Email Sending", False)
    use_real_emails = st.checkbox("Use real emails (needs 'email' column)", False)
    subject = st.text_input("Subject", "Counseling: Support available")
    body_template = st.text_area(
        "Body template",
        "Dear Student {student_id},\nAttendance: {attendance_pct}%, Test Avg: {avg_test_pct}%\nRegards"
    )

    import concurrent.futures
    import time

    def send_email(to_email, subject, body):
        """Send a single email and return (to_email, success, error_message)"""
        try:
            host = os.getenv("EMAIL_HOST")
            port = int(os.getenv("EMAIL_PORT", 587))
            user = os.getenv("EMAIL_USER")
            password = os.getenv("EMAIL_PASSWORD")
            from_addr = os.getenv("FROM_ADDRESS", user)

            if not all([host, port, user, password]):
                return (to_email, False, "SMTP credentials missing")

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = to_email
            msg.set_content(body)

            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, password)
                server.send_message(msg)

            return (to_email, True, None)
        except Exception as e:
            return (to_email, False, str(e))

    if enable_email and st.button("Send Emails"):
        if high_risk.empty:
            st.info("No high-risk students to send emails.")
        else:
            st.info("Sending emails asynchronously...")

            # Prepare email data
            email_tasks = []
            for _, row in high_risk.iterrows():
                student_id = row.get("student_id", "Unknown")
                to_email = row["email"] if use_real_emails and "email" in row else "ffnrnindian@gmail.com"
                body = body_template.format(
                    student_id=student_id,
                    attendance_pct=row.get("attendance_pct", "N/A"),
                    avg_test_pct=row.get("avg_test_pct", "N/A")
                )
                email_tasks.append((to_email, subject, body))

            sent_count = 0
            failed_list = []

            # Initialize progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(email_tasks)

            # Send emails in parallel threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(send_email, e[0], e[1], e[2]) for e in email_tasks]

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    to_email, success, error = future.result()
                    if success:
                        sent_count += 1
                        print(f"[SUCCESS] Email sent to {to_email}")
                    else:
                        failed_list.append((to_email, error))
                        print(f"[ERROR] Failed to send email to {to_email}: {error}")

                    # Update progress bar
                    progress = (i + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Sent: {sent_count}, Failed: {len(failed_list)}, Remaining: {total - (i+1)}")
                    time.sleep(0.05)  # small delay for UI update

            st.success(f"✅ Finished sending emails. Sent: {sent_count}, Failures: {len(failed_list)}")
            if failed_list:
                st.write("Failed recipients (first 10):", failed_list[:10])

# ---------------- YouTube Tutorial Section ----------------
import streamlit as st

st.write("---")

st.markdown(
    """
    <div style='text-align:center;'>
        <h3 style='color:#2E86C1;'>🎥 Watch our full tutorial on YouTube</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:center; margin-bottom:10px;'>
        <a href='https://www.youtube.com/watch?v=w5qt-TfBiKo' target='_blank'
           style='font-size:20px; color:#FF0000; text-decoration:none; font-weight:bold;'>
           ▶ Click here to watch on YouTube
        </a>
    </div>

    <!-- Custom-sized YouTube embed -->
    <div style='display:flex; justify-content:center;'>
        <iframe width="640" height="360"
                src="https://www.youtube.com/embed/w5qt-TfBiKo"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
        </iframe>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ---------------- Footer ----------------
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f5f5f5;
    color: #555;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #ddd;
}
</style>
<div class="footer">
  © 2025 GuardianAI | Developed by <b>GEN Z CODERS</b> | 💌 Contact: ffnrnindian@gmail.com
</div>
""", unsafe_allow_html=True)
st.write("---")
