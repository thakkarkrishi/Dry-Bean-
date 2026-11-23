
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
    color: #222222;
}

h1, h2, h3 {
    color: #245c43 !important;
}

p, label, li {
    color: #222222 !important;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #cccccc;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #333333 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #245c43 !important;
    font-size: 25px !important;
    font-weight: bold;
}

[data-testid="stSidebar"] {
    background-color: #eef4ef;
}

[data-testid="stSidebar"] * {
    color: #222222 !important;
}

.stButton button {
    background-color: #245c43;
    color: white !important;
    border-radius: 7px;
}

.stButton button p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")


# LOAD DATASET
@st.cache_data
def load_data():

    df = pd.read_csv(
        "D:/data science/dry bean dataset/drybean.csv"
    )

    for col in df.columns:

        if df[col].isnull().sum() > 0:

            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


# LOAD SAVED BEST MODEL
@st.cache_resource
def load_model():

    saved_data = joblib.load("dry_bean_model.joblib")

    return (
        saved_data["model"],
        saved_data["label_encoder"],
        saved_data["features"]
    )


try:
    df = load_data()

    best_model, le, features = load_model()

    X = df[features]

except FileNotFoundError:
    st.error(
        "Dataset or dry_bean_model.joblib file not found. "
        "Please check the file paths."
    )
    st.stop()


# GET MODEL NAME
if hasattr(best_model, "named_steps"):

    estimator = list(best_model.named_steps.values())[-1]

else:

    estimator = best_model

model_name = estimator.__class__.__name__

model_names = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
    "KNeighborsClassifier": "KNN",
    "SVC": "SVM"
}

model_name = model_names.get(model_name, model_name)


# SIDEBAR
st.sidebar.title("🌱 Dry Bean Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "EDA",
        "Model Performance",
        "Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dry Bean Classification")


# =====================================
# DASHBOARD
# =====================================

if page == "Dashboard":

    st.title("🌱 Dry Bean Classification")

    st.write("Machine Learning Classification Dashboard")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")

    c2.metric("Total Features", len(features))

    c3.metric("Bean Classes", df["Class"].nunique())

    c4.metric("Selected Model", model_name)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(8),
        use_container_width=True
    )

    st.subheader("Dry Bean Class Distribution")

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.countplot(
        data=df,
        x="Class",
        ax=ax,
        color="#4c956c"
    )

    ax.set_xlabel("Bean Class")
    ax.set_ylabel("Count")
    ax.set_title("Bean Class Distribution")

    ax.tick_params(
        axis="x",
        rotation=25,
        labelsize=9
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


# =====================================
# EDA
# =====================================

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    st.write(
        "Explore dataset structure, statistics and relationships."
    )

    st.divider()

    st.subheader("1. Dataset Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", df.shape[0])

    c2.metric("Total Columns", df.shape[1])

    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.divider()

    st.subheader("2. Dataset Preview")

    st.dataframe(
        df.head(8),
        use_container_width=True
    )

    st.divider()

    st.subheader("3. Statistical Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("4. Feature Distribution")

    feature = st.selectbox(
        "Select Feature",
        features
    )

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.histplot(
        df[feature],
        kde=True,
        ax=ax,
        color="#4c956c"
    )

    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)

    st.divider()

    st.subheader("5. Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        X.corr(),
        cmap="coolwarm",
        annot=False,
        linewidths=0.3,
        ax=ax
    )

    ax.set_title("Feature Correlation")
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)

    st.divider()

    st.subheader("6. Class Distribution")

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.countplot(
        data=df,
        x="Class",
        ax=ax,
        color="#4c956c"
    )

    ax.set_title("Dry Bean Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    ax.tick_params(
        axis="x",
        rotation=25,
        labelsize=9
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


# =====================================
# MODEL PERFORMANCE
# =====================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Information about the saved best classification model."
    )

    st.divider()

    st.subheader("Selected Best Model")

    c1, c2 = st.columns(2)

    c1.metric("Best Model", model_name)

    c2.metric("Total Features Used", len(features))

    st.success(
        f"{model_name} is loaded successfully from "
        "dry_bean_model.joblib."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:**", model_name)

    st.write("**Input Features:**", len(features))

    st.write("**Target Variable:** Class")

    st.write("**Classification Type:** Multiclass Classification")

    st.divider()

    st.subheader("Features Used by Model")

    st.dataframe(
        pd.DataFrame({
            "Feature Name": features
        }),
        use_container_width=True,
        hide_index=True
    )


# =====================================
# PREDICTION
# =====================================

elif page == "Prediction":

    st.title("🌱 Dry Bean Prediction")

    st.write(
        "Enter physical measurements to predict the dry bean class."
    )

    st.divider()

    st.subheader("Enter Feature Values")

    input_values = {}

    c1, c2 = st.columns(2)

    for i, feature in enumerate(features):

        min_value = float(df[feature].min())

        max_value = float(df[feature].max())

        mean_value = float(df[feature].mean())

        with c1 if i % 2 == 0 else c2:

            input_values[feature] = st.number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=mean_value,
                format="%.3f"
            )

    st.divider()

    if st.button(
        "Predict Dry Bean",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        # PREDICT BEAN CLASS
        prediction = best_model.predict(input_df)[0]

        # CONVERT NUMERIC PREDICTION INTO BEAN NAME
        if isinstance(prediction, (int, np.integer)):
            prediction = le.inverse_transform(
                [int(prediction)]
            )[0]

        # DISPLAY PREDICTED BEAN NAME
        st.markdown(
            f"""
            <div style="
                background-color: #dcfce7;
                color: #14532d;
                padding: 15px;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
                text-align: center;">
                Predicted Dry Bean: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

        # PREDICTION PROBABILITIES
        if hasattr(best_model, "predict_proba"):

            try:

                probabilities = best_model.predict_proba(
                    input_df
                )[0]

                probability_df = pd.DataFrame({
                    "Bean Class": le.classes_,
                    "Probability (%)": probabilities * 100
                })

                probability_df = probability_df.sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.subheader("Prediction Probabilities")

                st.dataframe(
                    probability_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )

                fig, ax = plt.subplots(figsize=(7, 3))

                sns.barplot(
                    data=probability_df,
                    x="Bean Class",
                    y="Probability (%)",
                    ax=ax,
                    color="#4c956c"
                )

                ax.set_title("Prediction Probabilities")
                ax.set_xlabel("Bean Class")
                ax.set_ylabel("Probability (%)")

                ax.tick_params(
                    axis="x",
                    rotation=25,
                    labelsize=8
                )

                plt.tight_layout()

                st.pyplot(fig, use_container_width=False)

                plt.close(fig)

            except (AttributeError, ValueError):
                pass
