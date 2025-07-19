import streamlit as st
import pandas as pd
import joblib

# Load the trained classification model
model = joblib.load("salary_predictor_model.pkl")  # Make sure this model file exists

# Set Streamlit app config
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Classification App")
st.markdown("This app predicts whether an employee's salary is **>100K** or **≤100K** based on input features.")

# -------------------------------------
# 🧠 Encoding Mappings (Match Model Training)
# -------------------------------------
gender_map = {'Male': 0, 'Female': 1}
education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
job_title_map = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "System Analyst": 2,
    "Project Manager": 3,
    "Business Analyst": 4,
    "DevOps Engineer": 5,
    "Database Administrator": 6,
    "Web Developer": 7,
    "Machine Learning Engineer": 8,
    "Network Engineer": 9
}

# -------------------------------------
# Sidebar: User Input
# -------------------------------------
st.sidebar.header("🧾 Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
education = st.sidebar.selectbox("Education Level", list(education_map.keys()))
job_title = st.sidebar.selectbox("Job Title", list(job_title_map.keys()))
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Display raw input
input_display_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [experience]
})
st.subheader("🔎 Input Summary")
st.dataframe(input_display_df)

# Encode input for prediction
encoded_input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_map[gender]],
    "Education Level": [education_map[education]],
    "Job Title": [job_title_map[job_title]],
    "Years of Experience": [experience]
})

# -------------------------------------
# 🔍 Predict Button
# -------------------------------------
if st.button("🔍 Predict Salary Class"):
    try:
        prediction = model.predict(encoded_input_df)
        st.success(f"🎯 Prediction: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# ----------------------------------------------------
# 📂 Batch Prediction Section
# ----------------------------------------------------
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with employee details", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(batch_df.head())

        # Check required columns
        required_cols = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
        if not all(col in batch_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in batch_df.columns]
            st.error(f"⚠️ Missing columns in CSV: {', '.join(missing)}")
        else:
            # Map categorical values
            batch_df["Gender"] = batch_df["Gender"].map(gender_map)
            batch_df["Education Level"] = batch_df["Education Level"].map(education_map)
            batch_df["Job Title"] = batch_df["Job Title"].map(job_title_map)

            if batch_df[required_cols].isnull().any().any():
                st.error("❌ Some values could not be encoded. Check for typos in categorical fields.")
            else:
                predictions = model.predict(batch_df[required_cols])
                batch_df["PredictedClass"] = predictions

                st.success("✅ Batch prediction complete!")
                st.dataframe(batch_df.head())

                # Download result
                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv,
                    file_name="salary_predictions.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"⚠️ Error processing the file: {e}")
