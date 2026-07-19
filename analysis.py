import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Page layout setup
st.set_page_config(page_title="Student AI Advisor Pro", layout="wide")

# Define our 6 predictive features
feature_cols = ["Math", "Science", "English", "History", "Art", "absences"]

# ==========================================
# 🔐 STUDENT LOGIN AUTHENTICATION SYSTEM
# ==========================================
# Mock student database mapped to system IDs
STUDENT_DB = {
    "ST101": {"name": "VIDUSHI", "Math": 34.0, "Science": 42.0, "English": 38.0, "History": 29.0, "Art": 45.0, "absences": 3},
    "ST102": {"name": "Priya Sharma", "Math": 48.0, "Science": 46.0, "English": 44.0, "History": 49.0, "Art": 38.0, "absences": 1},
    "ST103": {"name": "NEHRAJI", "Math": 22.0, "Science": 18.0, "English": 25.0, "History": 30.0, "Art": 28.0, "absences": 14},
    "STT104": {"name": "SHASHI", "Math": 41.0, "Science": 35.0, "English": 47.0, "History": 40.0, "Art": 42.0, "absences": 5}
}

if "logged_in_student" not in st.session_state:
    st.session_state.logged_in_student = None
if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []

# Login UI Frame
if st.session_state.logged_in_student is None:
    st.title("🔐 Student Performance Portal Access")
    st.write("Please authenticate using your assigned Student Identification Number to access predictions.")
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        student_id_input = st.text_input("Enter Student ID:", placeholder="e.g., STU001, STU002").strip().upper()
        if st.button("Log In"):
            if student_id_input in STUDENT_DB:
                st.session_state.logged_in_student = student_id_input
                st.success(f"Welcome back, {STUDENT_DB[student_id_input]['name']}!")
                st.rerun()
            else:
                st.error("Invalid Student ID. Use STU001, STU002, STU003, or STU004 for evaluation testing.")
    
    # Show available test IDs in an expander for ease of grading/testing
    with st.expander("💡 Portfolio Review: Available Test Student Profiles"):
        st.write("You can use any of the following valid IDs to preview different academic standing behaviors:")
        st.json(STUDENT_DB)
    st.stop() # Prevents the rest of the application dashboard from loading until validated

# Retrieve details for authenticated session profile object
current_student_id = st.session_state.logged_in_student
student_profile = STUDENT_DB[current_student_id]

# ==========================================
# ⚙️ SYSTEM SETTINGS CONFIGURATION
# ==========================================
st.sidebar.header(f"👤 Account: {student_profile['name']}")
st.sidebar.write(f"ID: {current_student_id}")
if st.sidebar.button("🔓 Log Out"):
    st.session_state.logged_in_student = None
    st.rerun()

st.sidebar.divider()
st.sidebar.header("⚙️ System Settings")
max_score_limit = st.sidebar.selectbox(
    "Select Grading Scale Base:",
    [20.0, 50.0, 100.0],
    index=1,
    format_func=lambda x: f"Out of {int(x)}"
)

# ==========================================
# DATA LOADING / AUTO-GENERATION
# ==========================================
use_mock_data = True

if os.path.exists("StudentData.csv"):
    try:
        df = pd.read_csv("StudentData.csv")
        if all(col in df.columns for col in feature_cols):
            use_mock_data = False
    except Exception:
        pass

if use_mock_data:
    np.random.seed(42)
    num_students = 100
    lower_bound = max_score_limit * 0.25
    mock_data = {sub: np.random.uniform(lower_bound, max_score_limit, num_students) for sub in feature_cols[:-1]}
    mock_data["absences"] = np.random.randint(0, 31, num_students)
    df = pd.DataFrame(mock_data)
    
    absence_weight = 0.35 if max_score_limit == 50.0 else (0.7 if max_score_limit == 100.0 else 0.15)
    df["G3"] = (
        (df["Math"] * 0.3) + (df["Science"] * 0.25) + (df["English"] * 0.2) + 
        (df["History"] * 0.15) + (df["Art"] * 0.1) - (df["absences"] * absence_weight)
    )
    df["G3"] = df["G3"].clip(0.0, max_score_limit)

# Calculate base percentages
subject_sum = df["Math"] + df["Science"] + df["English"] + df["History"] + df["Art"]
df["Percentage (%)"] = (((subject_sum / 5) / max_score_limit) * 100).round(1)

# Train AI
X = df[feature_cols]
y = df["G3"]
ai_model = LinearRegression()
ai_model.fit(X, y)

# ==========================================
# SIDEBAR PANEL: LIVE SLIDERS (PRE-POPULATED BY USER DATA)
# ==========================================
st.sidebar.divider()
st.sidebar.header("📝 Simulator Settings")
st.sidebar.write("Values initialized directly from your record files. Adjust to run 'What-If' scenarios.")

# Automatically scale initial data if system settings limits change dynamically
scale_factor = max_score_limit / 50.0  # Base records are written out of 50
init_math = min(max_score_limit, student_profile["Math"] * scale_factor)
init_science = min(max_score_limit, student_profile["Science"] * scale_factor)
init_english = min(max_score_limit, student_profile["English"] * scale_factor)
init_history = min(max_score_limit, student_profile["History"] * scale_factor)
init_art = min(max_score_limit, student_profile["Art"] * scale_factor)

input_math = st.sidebar.slider("Mathematics Score", 0.0, max_score_limit, init_math, 1.0)
input_science = st.sidebar.slider("Science Score", 0.0, max_score_limit, init_science, 1.0)
input_english = st.sidebar.slider("English Score", 0.0, max_score_limit, init_english, 1.0)
input_history = st.sidebar.slider("History Score", 0.0, max_score_limit, init_history, 1.0)
input_art = st.sidebar.slider("Art Score", 0.0, max_score_limit, init_art, 1.0)

st.sidebar.divider()
input_absences = st.sidebar.slider("Total Absences", min_value=0, max_value=80, value=student_profile["absences"])

st.sidebar.divider()
st.sidebar.header("🎯 Target Goal Setter")
target_grade = st.sidebar.slider(f"Set Target Final Grade:", min_value=0.0, max_value=max_score_limit, value=max_score_limit * 0.7, step=1.0)

# ==========================================
# MAIN DASHBOARD PAGE
# ==========================================
st.title("🎓 Multi-Subject Student Performance AI")
st.subheader(f"👋 Active Session Profile: {student_profile['name']} ({current_student_id})")
st.write("Review your authenticated record analytics and simulate metric goals dynamically below.")
st.divider()

# Projections logic calculations
user_inputs = [[input_math, input_science, input_english, input_history, input_art, input_absences]]
predicted_grade = max(0.0, min(max_score_limit, ai_model.predict(user_inputs)[0]))
predicted_percentage = (predicted_grade / max_score_limit) * 100

current_avg = (input_math + input_science + input_english + input_history + input_art) / 5
user_percentage = (current_avg / max_score_limit) * 100

# Metric blocks output
col1, col2, col3 = st.columns(3)

with col1:
    st.write("### 🔮 AI Future Projection")
    passing_threshold = max_score_limit * 0.5
    metric_value = f"{predicted_grade:.1f} / {int(max_score_limit)} ({predicted_percentage:.1f}%)"
    if predicted_grade >= passing_threshold:
        st.metric(label="Predicted Final Year-End Grade", value=metric_value, delta="PROJECTED PASS")
    else:
        st.metric(label="Predicted Final Year-End Grade", value=metric_value, delta="PROJECTED RISK", delta_color="inverse")

with col2:
    st.write("### 📊 Current Mid-Term Status")
    st.metric(label="Average of Entered Marks", value=f"{current_avg:.1f} / {int(max_score_limit)}")

with col3:
    st.write("### 💯 Current Academic Weight")
    st.metric(label="Current Percentage Score", value=f"{user_percentage:.1f}%")

st.divider()

# Snapshot trigger interface module
st.write("### 📸 Performance Snapshot Logger")
st.write("Save your alternative slider scenarios to track custom academic optimization paths over time.")

label_input = st.text_input("Enter a Record Title:", placeholder="e.g., Target Scenario 1, Current Base Metric Profile")

if st.button("💾 Save Current Evaluation as History Entry"):
    if label_input.strip() == "":
        st.error("Please enter a record title before saving your evaluation snapshot.")
    else:
        snapshot = {
            "StudentID": current_student_id,
            "Assessment Label": label_input,
            "Math": f"{input_math:.1f}",
            "Science": f"{input_science:.1f}",
            "English": f"{input_english:.1f}",
            "History": f"{input_history:.1f}",
            "Art": f"{input_art:.1f}",
            "Absences": input_absences,
            "Mid-Term Status (Avg)": f"{current_avg:.1f} / {int(max_score_limit)}",
            "Current Academic Weight (%)": f"{user_percentage:.1f}%",
            "AI Future Prediction": metric_value
        }
        st.session_state.evaluation_history.append(snapshot)
        st.success("Snapshot successfully appended to the active student session ledger!")

# Display log if populated
if st.session_state.evaluation_history:
    st.write("#### 🕒 Saved Progress & Past Performance History")
    # Filter the layout log view so students only track entries tied to their unique validated ID profile
    full_history_df = pd.DataFrame(st.session_state.evaluation_history)
    filtered_history_df = full_history_df[full_history_df["StudentID"] == current_student_id]
    
    if not filtered_history_df.empty:
        st.dataframe(filtered_history_df.drop(columns=["StudentID"]), use_container_width=True)
    
    if st.button("🗑️ Clear My Session Log"):
        st.session_state.evaluation_history = [entry for entry in st.session_state.evaluation_history if entry["StudentID"] != current_student_id]
        st.rerun()

st.divider()

st.write("### 📊 Goal Gap Analysis")
progress_percentage = min(1.0, max(0.0, predicted_grade / target_grade)) if target_grade > 0 else 0.0
st.progress(progress_percentage)

if predicted_grade >= target_grade:
    st.success(f"🥳 Excellent! The AI predicts you will meet or exceed your target benchmark goal of {target_grade}/{int(max_score_limit)}.")
else:
    gap = target_grade - predicted_grade
    st.info(f"📈 **Goal Gap:** Projected to fall short of your target profile threshold by **{gap:.1f} grade points**.")