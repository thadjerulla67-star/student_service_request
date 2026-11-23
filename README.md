# Student Service Request Monitoring System

Streamlit implementation for Section B, reusing the submitted midterm Decision Tree model.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

The app opens in your browser. The included CSV is for local demonstration. Use only synthetic or properly authorized data, especially for public deployment.

## Project structure

- `app.py` — Streamlit navigation and page layout
- `modules/data_processing.py` — schema validation and feature preparation
- `modules/exploratory_analysis.py` — interactive EDA charts
- `modules/prediction.py` — cached saved-model loading and inference
- `modules/dashboard.py` — KPIs and risk review queue
- `models/trained_model.joblib` — fitted preprocessing + Decision Tree pipeline
- `models/model_metrics.json` — held-out test metrics and feature metadata
- `assets/use_case_diagram.png` — administrator use-case diagram
- `data/student_service_requests.csv` — uploaded demonstration dataset

## Midterm continuity

The fitted pipeline was trained from the uploaded midterm notebook using:
- Eligible population: records with `Status == Completed`, non-null `Processing_Days`, and non-null `SLA_Days`
- Target: `SLA_Breached = 1` when `Processing_Days > SLA_Days`, otherwise `0`
- Features: `Service_Type`, `Requester_Type`, `Requested_Channel`, `Priority`, `Assigned_Office`, `Queue_Length`, `Required_Documents_Complete`, `SLA_Days`, `Submission_Weekday`, `Queue_Category`
- Split: 80/20, `random_state=42`, stratified
- Model: `DecisionTreeClassifier(random_state=42)` inside the same one-hot preprocessing pipeline

`Processing_Days`, `Status`, `Reopened`, and `Satisfaction_Score` are not inference features. The app derives weekday and queue category from submission-time values.

## Test results from the reproduced midterm split

See `models/model_metrics.json` for exact metrics and confusion matrix. The application displays these results on the Prediction page.

## Deployment to Streamlit Community Cloud

1. Create a GitHub repository and upload the project files.
2. Do not publish real student records, passwords, or other confidential information. Remove `data/student_service_requests.csv` if it contains real or sensitive records; the app will then require a CSV upload, so update the default-data loader to handle the missing file or add a clearly synthetic demo dataset.
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set `app.py` as the main file.
5. Deploy, open the app URL, and inspect the deployment logs if an error occurs.

## Important limitations

- This is a classroom decision-support prototype, not a production student-records system.
- The model was trained on completed historical requests. Predictions for pending or newly submitted requests are estimates and may not generalize.
- Model probabilities are not guarantees or proof of causation.
- Authorized staff must review context and make the final decision. A high-risk prediction must never automatically deny a student service.
