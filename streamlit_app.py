import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Dataset Storyteller",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "token" not in st.session_state:
    st.session_state["token"] = None

if "result" not in st.session_state:
    st.session_state["result"] = None


# ============================================================
# HEADER
# ============================================================

st.title("📊 AI Dataset Storyteller")
st.caption(
    "Upload a dataset and let AI turn it into an understandable story."
)


# ============================================================
# SIDEBAR LOGIN
# ============================================================

st.sidebar.header("🔐 Account")

email = st.sidebar.text_input(
    "Email"
)

password = st.sidebar.text_input(
    "Password",
    type="password"
)


if st.sidebar.button("Login"):

    try:

        response = requests.post(
            f"{API_URL}/login",
            data={
                "username": email,
                "password": password
            }
        )

        if response.status_code == 200:

            data = response.json()

            st.session_state["token"] = (
                data["access_token"]
            )

            st.session_state["logged_in"] = True

            st.sidebar.success(
                "Login successful!"
            )

            st.rerun()

        else:

            st.sidebar.error(
                "Incorrect email or password."
            )

    except requests.exceptions.ConnectionError:

        st.sidebar.error(
            "FastAPI server is not running."
        )


# ============================================================
# LOGOUT
# ============================================================

if st.session_state["logged_in"]:

    if st.sidebar.button("Logout"):

        st.session_state["logged_in"] = False
        st.session_state["token"] = None
        st.session_state["result"] = None

        st.rerun()


# ============================================================
# LOGIN REQUIRED
# ============================================================

if not st.session_state["logged_in"]:

    st.info(
        "👈 Login from the sidebar to analyze a dataset."
    )

    st.stop()


st.success("You are logged in.")


# ============================================================
# UPLOAD SECTION
# ============================================================

st.divider()

st.header("📁 Analyze Your Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button(
        "🚀 Analyze Dataset",
        type="primary"
    ):

        token = st.session_state["token"]

        headers = {
            "Authorization": f"Bearer {token}"
        }

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        }

        try:

            with st.spinner(
                "Analyzing your dataset..."
            ):

                response = requests.post(
                    f"{API_URL}/upload",
                    headers=headers,
                    files=files
                )

            if response.status_code == 200:

                result = response.json()

                st.session_state["result"] = result

                st.success(
                    "✅ Dataset analyzed successfully!"
                )

            else:

                st.error(
                    f"Upload failed: {response.text}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Make sure the server is running."
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )


# ============================================================
# GET RESULT
# ============================================================

result = st.session_state.get(
    "result"
)


if result is None:

    st.info(
        "Upload a CSV and click **Analyze Dataset** "
        "to see the results."
    )

    st.stop()


# ============================================================
# BASIC DATA
# ============================================================

num_rows = result.get(
    "num_rows",
    0
)

num_columns = result.get(
    "num_columns",
    0
)

column_names = result.get(
    "column_names",
    []
)

cleaning_report = result.get(
    "cleaning_report",
    {}
)

describe_data = result.get(
    "describe",
    {}
)

correlation = result.get(
    "correlation",
    {}
)

categorical_summary = result.get(
    "categorical_summary",
    {}
)

group_analysis = result.get(
    "group_analysis",
    {}
)

key_findings = result.get(
    "key_findings",
    []
)

anomalies = result.get(
    "anomalies",
    {}
)

residual_outliers = result.get(
    "residual_outliers",
    {}
)

ai_narrative = result.get(
    "ai_narrative"
)

fallback_summary = result.get(
    "fallback_summary",
    ""
)

used_fallback = result.get(
    "used_fallback",
    False
)

history_id = result.get(
    "history_id"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.header("📊 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# ROWS
# ------------------------------------------------------------

with col1:

    st.metric(
        "Rows",
        num_rows
    )


# ------------------------------------------------------------
# COLUMNS
# ------------------------------------------------------------

with col2:

    st.metric(
        "Columns",
        num_columns
    )


# ------------------------------------------------------------
# ANOMALIES
# ------------------------------------------------------------

with col3:

    if isinstance(
        anomalies,
        dict
    ):

        anomaly_count = anomalies.get(
            "anomaly_count",
            0
        )

    else:

        anomaly_count = 0

    st.metric(
        "Anomalies",
        anomaly_count
    )


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

with col4:

    missing_values = 0

    if isinstance(
        cleaning_report,
        dict
    ):

        missing_before = cleaning_report.get(
            "missing_values_before",
            {}
        )

        if isinstance(
            missing_before,
            dict
        ):

            for value in missing_before.values():

                if isinstance(
                    value,
                    (int, float)
                ):

                    missing_values += value

    st.metric(
        "Missing Values",
        missing_values
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

if column_names:

    with st.expander("📋 Dataset Columns"):

        st.write(
            column_names
        )


# ============================================================
# AI DATA STORY
# ============================================================

st.divider()

st.header("🤖 AI Dataset Story")

if ai_narrative:

    st.markdown(
        ai_narrative
    )

elif fallback_summary:

    st.write(
        fallback_summary
    )

else:

    st.info(
        "No AI narrative was generated."
    )


if used_fallback:

    st.caption(
        "ℹ️ The system used the analytical fallback summary."
    )


# ============================================================
# KEY FINDINGS
# ============================================================

st.divider()

st.header("🔎 Key Findings")


if key_findings:

    for index, finding in enumerate(
        key_findings,
        start=1
    ):

        if isinstance(
            finding,
            dict
        ):

            title = finding.get(
                "title",
                f"Finding {index}"
            )

            description = finding.get(
                "description",
                ""
            )

            st.markdown(
                f"### {index}. {title}"
            )

            st.write(
                description
            )

        else:

            st.markdown(
                f"### {index}. Finding"
            )

            st.write(
                finding
            )

else:

    st.info(
        "No key findings were generated."
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

st.divider()

st.header("📈 Important Relationships")


if correlation:

    correlation_rows = []

    if isinstance(
        correlation,
        dict
    ):

        for column1, values in correlation.items():

            if not isinstance(
                values,
                dict
            ):

                continue

            for column2, value in values.items():

                if column1 == column2:
                    continue

                if isinstance(
                    value,
                    (int, float)
                ):

                    correlation_rows.append(
                        {
                            "Variable 1": column1,
                            "Variable 2": column2,
                            "Correlation": value
                        }
                    )


    if correlation_rows:

        correlation_rows.sort(
            key=lambda x: abs(
                x["Correlation"]
            ),
            reverse=True
        )

        displayed_pairs = set()

        final_correlations = []

        for row in correlation_rows:

            pair = frozenset(
                [
                    row["Variable 1"],
                    row["Variable 2"]
                ]
            )

            if pair in displayed_pairs:
                continue

            displayed_pairs.add(
                pair
            )

            final_correlations.append(
                row
            )


        correlation_df = pd.DataFrame(
            final_correlations
        )

        st.dataframe(
            correlation_df,
            use_container_width="strech",
            hide_index=True
        )


        # --------------------------------------------------------
        # CORRELATION BAR CHART
        # --------------------------------------------------------

        if len(final_correlations) > 0:

            chart_df = correlation_df.head(
                min(10, len(correlation_df))
            ).copy()

            chart_df["Relationship"] = (
                chart_df["Variable 1"]
                + " ↔ "
                + chart_df["Variable 2"]
            )

            chart_df = chart_df.sort_values(
                "Correlation"
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                chart_df["Relationship"],
                chart_df["Correlation"]
            )

            ax.set_xlabel(
                "Correlation"
            )

            ax.set_title(
                "Strongest Variable Relationships"
            )

            ax.axvline(
                0
            )

            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(fig)

    else:

        st.info(
            "Correlation data is available but "
            "could not be displayed."
        )

else:

    st.info(
        "No correlation data available."
    )


# ============================================================
# CATEGORICAL INSIGHTS
# ============================================================

if categorical_summary:

    st.divider()

    st.header("🏷️ Categorical Insights")


    for column, data in categorical_summary.items():

        st.subheader(
            column
        )

        if isinstance(
            data,
            dict
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Unique Values",
                    data.get(
                        "unique_values",
                        "N/A"
                    )
                )

            with col2:

                st.metric(
                    "Most Common",
                    data.get(
                        "top_value",
                        "N/A"
                    )
                )

            with col3:

                st.metric(
                    "Frequency",
                    data.get(
                        "top_value_count",
                        "N/A"
                    )
                )

        else:

            st.write(
                data
            )


# ============================================================
# ANOMALY ANALYSIS
# ============================================================

st.divider()

st.header("🚨 Anomaly Analysis")


if isinstance(
    anomalies,
    dict
):

    anomaly_count = anomalies.get(
        "anomaly_count",
        0
    )

    anomaly_percentage = anomalies.get(
        "anomaly_percentage",
        0
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Unusual Records",
            anomaly_count
        )

    with col2:

        if isinstance(
            anomaly_percentage,
            (int, float)
        ):

            st.metric(
                "Anomaly Percentage",
                f"{anomaly_percentage:.2f}%"
            )

        else:

            st.metric(
                "Anomaly Percentage",
                anomaly_percentage
            )


    if anomaly_count:

        st.warning(
            "The analysis identified statistically "
            "unusual observations in the dataset."
        )

    else:

        st.success(
            "No significant anomalies were detected."
        )

else:

    st.info(
        "No anomaly information available."
    )


# ============================================================
# RESIDUAL OUTLIERS
# ============================================================

if residual_outliers:

    st.divider()

    st.header("📉 Residual Outliers")

    st.json(
        residual_outliers
    )


# ============================================================
# GROUP ANALYSIS
# ============================================================

if group_analysis:

    st.divider()

    st.header("🌎 Group Analysis")


    for column, groups in group_analysis.items():

        st.subheader(
            column
        )

        if isinstance(
            groups,
            dict
        ):

            rows = []

            for group_name, values in groups.items():

                if isinstance(
                    values,
                    dict
                ):

                    row = {
                        "Group": group_name
                    }

                    for metric, value in values.items():

                        row[metric] = value

                    rows.append(
                        row
                    )

                else:

                    rows.append(
                        {
                            "Group": group_name,
                            "Value": values
                        }
                    )


            if rows:

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width="strech",
                    hide_index=True
                )

        else:

            st.write(
                groups
            )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

if describe_data:

    st.divider()

    st.header("📐 Descriptive Statistics")

    try:

        describe_df = pd.DataFrame(
            describe_data
        )

        st.dataframe(
            describe_df,
            use_container_width="strech"
        )

    except Exception:

        st.json(
            describe_data
        )


# ============================================================
# CLEANING REPORT
# ============================================================

if cleaning_report:

    st.divider()

    st.header("🧹 Data Cleaning Report")

    st.json(
        cleaning_report
    )


# ============================================================
# ASK AI ABOUT DATASET
# ============================================================

if history_id:

    st.divider()

    st.header("💬 Ask About Your Dataset")

    question = st.text_input(
        "Ask a question about this dataset",
        placeholder="Example: Which variables are most strongly related?"
    )


    if st.button(
        "Ask AI"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            token = st.session_state["token"]

            headers = {
                "Authorization": f"Bearer {token}"
            }

            payload = {
                "history_id": history_id,
                "question": question
            }

            try:

                with st.spinner(
                    "Thinking..."
                ):

                    response = requests.post(
                        f"{API_URL}/ask",
                        headers=headers,
                        json=payload
                    )


                if response.status_code == 200:

                    answer = response.json()

                    st.success(
                        "AI Answer"
                    )

                    st.write(
                        answer.get(
                            "answer",
                            "No answer returned."
                        )
                    )

                else:

                    st.error(
                        f"Could not answer question: "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI."
                )