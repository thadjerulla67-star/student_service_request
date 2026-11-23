
import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.data_processing import validate_dataset, prepare_features, make_target
from modules.exploratory_analysis import service_volume_chart, status_chart, queue_chart, office_breach_chart
from modules.prediction import load_model, load_metrics, predict_frame, predict_one
from modules.dashboard import render_dashboard

st.set_page_config(page_title="Student Service Request Monitoring System", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
.block-container {padding-top:1.5rem; padding-bottom:2rem;}
[data-testid="stSidebar"] {background-color:#edf2f7;}
.hero {padding:1.4rem 1.6rem; border-radius:14px; background:linear-gradient(120deg,#17365d,#315b85); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#e7eef7; margin:.35rem 0 0 0;}
.small-note {color:#5c677d; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_default_data():
    return pd.read_csv(ROOT / "data" / "student_service_requests.csv")

if "working_df" not in st.session_state:
    st.session_state.working_df = load_default_data()

df = st.session_state.working_df.copy()
valid, message = validate_dataset(df)
menu = st.sidebar.selectbox(
    "NAVIGATION",
    ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Student Service Request Monitoring System")
st.sidebar.caption("Section B | Systems Analysis and Design")
st.sidebar.caption("Model: Decision Tree Classifier")

if menu == "Home":
    st.markdown('<div class="hero"><h1>Student Service Request Monitoring System</h1><p>Monitor service demand, review SLA risk, and support timely staff decisions.</p></div>', unsafe_allow_html=True)
    st.write("This application reuses the Decision Tree model from the submitted midterm notebook. It provides dataset validation, exploratory analysis, advisory predictions, an administrator dashboard, and system-design documentation.")
    a,b,c = st.columns(3)
    a.metric("Dataset Records", f"{len(df):,}")
    b.metric("Pending Requests", f"{(df['Status']=='Pending').sum():,}")
    c.metric("Completed Requests", f"{(df['Status']=='Completed').sum():,}")
    st.markdown("### System modules")
    cards = st.columns(3)
    cards[0].info("**Dataset**\\n\\nUpload a CSV, validate its schema, and inspect missing or duplicate values.")
    cards[1].info("**Data Analysis**\\n\\nExplore request volume, status, queue length, and observed SLA performance.")
    cards[2].info("**Prediction**\\n\\nEstimate the likelihood of SLA breach using the saved midterm pipeline.")
    cards2 = st.columns(3)
    cards2[0].info("**Dashboard**\\n\\nReview operational KPIs and the advisory risk queue.")
    cards2[1].info("**System Design**\\n\\nReview actors, requirements, use case, and decision workflow.")
    cards2[2].success("**Human-centered use**\\n\\nStaff make the final decision after reviewing each flagged request.")
    st.caption("The bundled CSV is provided for demonstration and local testing. Replace it only with synthetic or properly authorized data.")

elif menu == "Dataset":
    st.title("Dataset Upload and Validation")
    st.write("Upload the student service request CSV. The expected schema follows the midterm dataset.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            ok, msg = validate_dataset(uploaded_df)
            if not ok:
                st.error(msg)
            elif uploaded_df.empty:
                st.error("The uploaded CSV has no rows.")
            else:
                st.session_state.working_df = uploaded_df
                df = uploaded_df.copy()
                st.success("CSV uploaded and schema validated.")
        except Exception as exc:
            st.error(f"Unable to read this CSV: {exc}")
    ok, msg = validate_dataset(df)
    if ok:
        st.success(msg)
        x1,x2,x3 = st.columns(3)
        x1.metric("Rows", len(df))
        x2.metric("Columns", len(df.columns))
        x3.metric("Duplicate Request IDs", int(df["Request_ID"].duplicated().sum()))
        st.markdown("#### Data preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        st.markdown("#### Missing values")
        nulls = df.isna().sum().rename("Missing Values").to_frame()
        nulls["Missing %"] = (nulls["Missing Values"] / len(df) * 100).round(1)
        st.dataframe(nulls, use_container_width=True)
        st.markdown("#### Data types")
        st.dataframe(df.dtypes.astype(str).rename("Data Type").to_frame(), use_container_width=True)
        st.caption("Missing Processing_Days and Satisfaction_Score values are expected for some open or pending requests. They are not used as prediction inputs.")
    else:
        st.error(msg)

elif menu == "Data Analysis":
    st.title("Exploratory Data Analysis")
    ok,msg = validate_dataset(df)
    if not ok:
        st.error(msg)
    else:
        completed = df[(df["Status"]=="Completed") & df["Processing_Days"].notna() & df["SLA_Days"].notna()].copy()
        completed["SLA_Breached"] = (completed["Processing_Days"] > completed["SLA_Days"]).astype(int)
        st.markdown("#### Descriptive statistics")
        numeric = df.select_dtypes(include="number")
        st.dataframe(numeric.describe().T, use_container_width=True)
        if "SLA_Breached" in completed and len(completed):
            target_counts = completed["SLA_Breached"].map({0:"No breach",1:"Breach"}).value_counts().rename_axis("Target").reset_index(name="Records")
            st.plotly_chart(px.pie(target_counts, names="Target", values="Records", title="Historical SLA Breach Class Distribution", hole=.4), use_container_width=True)
        col1,col2 = st.columns(2)
        with col1:
            st.plotly_chart(service_volume_chart(df), use_container_width=True)
        with col2:
            st.plotly_chart(status_chart(df), use_container_width=True)
        col3,col4 = st.columns(2)
        with col3:
            st.plotly_chart(queue_chart(df), use_container_width=True)
        with col4:
            chart = office_breach_chart(df)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No completed records are available for office-level SLA analysis.")
        st.caption("Observed SLA breach rates are calculated only for completed records with recorded Processing_Days and SLA_Days. Pending records are not labeled as historical breaches.")

elif menu == "Prediction":
    st.title("SLA Breach Prediction")
    st.write("Enter submission-time information. Do not enter processing outcomes or other information learned after the request is submitted.")
    metrics = load_metrics()
    categories = {
        "Service_Type": sorted(df["Service_Type"].dropna().astype(str).unique().tolist()),
        "Requester_Type": sorted(df["Requester_Type"].dropna().astype(str).unique().tolist()),
        "Requested_Channel": sorted(df["Requested_Channel"].dropna().astype(str).unique().tolist()),
        "Priority": sorted(df["Priority"].dropna().astype(str).unique().tolist()),
        "Assigned_Office": sorted(df["Assigned_Office"].dropna().astype(str).unique().tolist()),
        "Required_Documents_Complete": sorted(df["Required_Documents_Complete"].dropna().astype(str).unique().tolist()),
        "Submission_Weekday": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        "Queue_Category": ["Low","Medium","High"]
    }
    with st.form("prediction_form"):
        c1,c2 = st.columns(2)
        with c1:
            service = st.selectbox("Service Type", categories["Service_Type"])
            requester = st.selectbox("Requester Type", categories["Requester_Type"])
            channel = st.selectbox("Requested Channel", categories["Requested_Channel"])
            priority = st.selectbox("Priority", categories["Priority"])
            office = st.selectbox("Assigned Office", categories["Assigned_Office"])
        with c2:
            docs = st.selectbox("Required Documents Complete", categories["Required_Documents_Complete"])
            queue = st.number_input("Queue Length", min_value=0, max_value=10000, value=8, step=1)
            sla = st.number_input("SLA Days", min_value=1, max_value=365, value=5, step=1)
            weekday = st.selectbox("Submission Weekday", categories["Submission_Weekday"])
        submitted = st.form_submit_button("Predict SLA Breach Risk", type="primary")
    if submitted:
        qcat = "Low" if queue <= 5 else ("Medium" if queue <= 15 else "High")
        values = {
            "Service_Type":service, "Requester_Type":requester,
            "Requested_Channel":channel, "Priority":priority,
            "Assigned_Office":office, "Queue_Length":int(queue),
            "Required_Documents_Complete":docs, "SLA_Days":int(sla),
            "Submission_Weekday":weekday, "Queue_Category":qcat
        }
        try:
            label, probability = predict_one(values)
            if probability is not None:
                st.metric("Estimated SLA Breach Probability", f"{probability:.1%}")
                st.progress(min(max(probability,0.0),1.0))
            if label == 1:
                st.warning("Model classification: SLA Breach Risk")
            else:
                st.success("Model classification: No SLA Breach Risk")
            st.write("**Suggested follow-up:** Staff should check the current workload, confirm document completeness, and review whether additional support or queue prioritization is appropriate.")
            st.caption("This is an advisory prediction based on historical patterns. It is not a guarantee and must not be used alone to deny, reject, or deprioritize student services.")
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
    with st.expander("Model evaluation from the midterm test split"):
        st.json({k:metrics[k] for k in ["accuracy","precision","recall","f1","majority_baseline","confusion_matrix","training_rows","testing_rows"]})

elif menu == "Dashboard":
    st.title("Administrator Decision-Support Dashboard")
    ok,msg = validate_dataset(df)
    if not ok:
        st.error(msg)
    else:
        try:
            with st.spinner("Scoring current requests with the saved model..."):
                predictions = predict_frame(df)
            render_dashboard(df, predictions)
        except Exception as exc:
            st.error(f"Dashboard prediction failed: {exc}")

elif menu == "System Design":
    st.title("System Design and SAD Integration")
    st.markdown("### Actors")
    st.write("- **Administrator / Student Services Staff:** monitors service volume and SLA performance, reviews flagged requests and queue status, and records final operational decisions.")
    st.markdown("### Functional requirements")
    st.write("1. The system shall display SLA breach rates by assigned office so administrators can identify offices requiring operational attention.")
    st.caption("Evidence from the submitted analysis: Admissions had the highest observed breach rate at approximately 48.3%.")
    st.write("2. The system shall display request volume by service type so administrators can identify high-demand services.")
    st.caption("Evidence: Scholarship Inquiry, ID Card, and IT Support each had 55 requests in the submitted dataset.")
    st.write("3. The system shall identify requests with characteristics associated with higher predicted SLA-breach risk and present them to administrators for review.")
    st.write("4. The system shall validate uploaded CSV files and report missing required columns, missing values, and duplicate request identifiers.")
    st.write("5. The system shall display model evaluation results and advisory predictions to authorized staff.")
    st.markdown("### Non-functional requirements")
    st.write("- **Usability:** Provide clear navigation, readable charts, and understandable risk labels.")
    st.write("- **Security and privacy:** Restrict access to authorized staff and use synthetic or properly authorized data for public demonstrations.")
    st.write("- **Reliability:** Validate input schemas, handle invalid files, and load a consistent saved model pipeline.")
    st.markdown("### Administrator use-case diagram")
    diagram = ROOT / "assets" / "use_case_diagram.png"
    if diagram.exists():
        st.image(str(diagram), caption="Administrator use cases supplied with the project", use_container_width=True)
    st.markdown("### Analytical decision workflow")
    st.code("CSV / request records\\n    ↓\\nSchema and data validation\\n    ↓\\nSubmission-time feature preparation\\n    ↓\\nSaved preprocessing + Decision Tree pipeline\\n    ↓\\nPredicted SLA-breach class / probability\\n    ↓\\nAdministrator reviews risk, queue, documents, and service context\\n    ↓\\nAuthorized staff records the final decision", language="text")
    st.markdown("### Model details and limitations")
    m = load_metrics()
    st.write(f"Model: **{m['model_name']}** | Split: **80% training / 20% testing** | Random state: **{m['random_state']}**.")
    st.write(f"Eligible historical records: **{m['eligible_rows']}** completed requests. Pending and cancelled requests were excluded from model training because their final processing outcome is not available.")
    st.warning("The model's measured performance is modest. Predictions support human triage only and do not automatically approve, reject, or deny student services.")
