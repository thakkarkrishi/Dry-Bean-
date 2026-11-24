import os
import warnings
warnings.filterwarnings("ignore")

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Machine Learning Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Dry Bean Classification & EDA",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (CSS)
# ==========================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff !important;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .main-header p {
        color: #d8f3dc !important;
        margin-bottom: 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #2d6a4f;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .algo-header {
        background-color: #f1f8f4;
        border-left: 5px solid #2d6a4f;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 18px;
    }
    .algo-header h3 {
        margin: 0;
        color: #1b4332 !important;
    }
    .algo-header p {
        margin: 5px 0 0 0;
        color: #4b5563 !important;
        font-size: 0.95rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        border: 2px solid #52b788;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-title {
        font-size: 15px;
        font-weight: 600;
        color: #1b4332;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-result {
        font-size: 32px;
        font-weight: 800;
        color: #081c15;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Seaborn & Matplotlib general theme
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = {}

# ==========================================
# DATA LOADING & PREPARATION
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_path="drybean.csv"):
    if not os.path.exists(file_path):
        return None, None, None, None, None
    df = pd.read_csv(file_path)
    
    # Handle missing values if any exist
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
                
    features = [c for c in df.columns if c != "Class"]
    X = df[features]
    y_raw = df["Class"]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    
    return df, X, y, features, class_names, le

df, X, y, feature_names, class_names, label_encoder = load_and_preprocess_data("drybean.csv")

if df is None:
    st.error("Error: Could not find 'drybean.csv'. Please make sure the file is located in the working directory.")
    st.stop()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551462147-ff29053bfc14?auto=format&fit=crop&w=400&q=80", caption="Dry Bean Classification", use_container_width=True)
    st.title("⚙️ Global Settings")
    
    test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05, format="%.2f")
    random_seed = st.number_input("Random Seed (Reproducibility)", min_value=0, max_value=999, value=42, step=1)
    stratify_split = st.checkbox("Stratified Split (Preserve Class Ratio)", value=True)
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Summary")
    st.write(f"• **Total Samples:** {len(df):,}")
    st.write(f"• **Features:** {len(feature_names)}")
    st.write(f"• **Classes:** {len(class_names)}")
    st.write(f"• **Missing Values:** {df.isnull().sum().sum()}")
    st.markdown("---")
    st.caption("Developed with Streamlit & Scikit-Learn 🌱")

# Train/Test Split
strat = y if stratify_split else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=strat
)

# Helper function to evaluate and save model in session state
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    start_time = time.time()
    model.fit(X_tr, y_tr)
    duration = time.time() - start_time
    
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    eval_dict = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "time": duration,
        "y_pred": y_pred,
        "cm": cm,
        "report": report
    }
    st.session_state["trained_models"][name] = eval_dict
    return eval_dict

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"}
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Predicted Bean Variety", fontsize=11, labelpad=8)
    ax.set_ylabel("True Bean Variety", fontsize=11, labelpad=8)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig

# ==========================================
# MAIN APP HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>🌱 Dry Bean Classification & EDA Studio</h1>
    <p>Perform comprehensive Exploratory Data Analysis and train multiple ML algorithms (SVM, Naive Bayes, Decision Tree, Logistic Regression, etc.) with real-time evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Top Metric Row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-value">13,611</div><div class="metric-label">Dataset Records</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-value">16</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-value">7</div><div class="metric-label">Bean Varieties</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(X_train):,}</div><div class="metric-label">Training Samples</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(X_test):,}</div><div class="metric-label">Testing Samples</div></div>', unsafe_allow_html=True)

st.write("")

# ==========================================
# APPLICATION TABS
# ==========================================
tab_eda, tab_svm, tab_nb, tab_dt, tab_lr, tab_rf, tab_knn, tab_compare, tab_predict = st.tabs([
    "📊 Basic EDA",
    "⚡ SVM",
    "🧪 Naive Bayes",
    "🌳 Decision Tree",
    "📈 Logistic Regression",
    "🌲 Random Forest",
    "🎯 KNN",
    "🏆 Model Comparison",
    "🔮 Live Bean Predictor"
])

# ==============================================================================
# TAB 1: BASIC EDA
# ==============================================================================
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")
    st.caption("Analyze structure, distributions, correlations, and class balance of the Dry Bean dataset.")
    
    eda_sub1, eda_sub2, eda_sub3, eda_sub4 = st.tabs([
        "📋 Dataset Overview",
        "🎯 Class Distribution",
        "📈 Feature Distributions",
        "🔗 Correlations & Relationships"
    ])
    
    with eda_sub1:
        st.markdown("#### Sample Records")
        n_rows = st.slider("Rows to preview", min_value=5, max_value=50, value=8, step=5)
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe().T.round(3), use_container_width=True)
        
        st.markdown("#### Feature Data Types & Null Values")
        info_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(info_df, use_container_width=True)
        
    with eda_sub2:
        st.markdown("#### Bean Varieties Distribution")
        col_c1, col_c2 = st.columns([1.8, 1.2])
        
        class_counts = df["Class"].value_counts().reset_index()
        class_counts.columns = ["Bean Variety", "Count"]
        class_counts["Percentage (%)"] = (class_counts["Count"] / len(df) * 100).round(2)
        
        with col_c1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            palette = sns.color_palette("Greens_r", n_colors=len(class_counts))
            sns.barplot(
                data=class_counts,
                x="Bean Variety",
                y="Count",
                palette=palette,
                ax=ax
            )
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height):,}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 4),
                            textcoords='offset points')
            ax.set_title("Bean Variety Class Counts", fontsize=12, fontweight='bold', pad=12)
            ax.set_ylabel("Number of Samples", fontsize=10)
            ax.set_xlabel("Dry Bean Class", fontsize=10)
            plt.xticks(rotation=25)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        with col_c2:
            st.markdown("##### Distribution Details")
            st.dataframe(class_counts, use_container_width=True, hide_index=True)
            most_common = class_counts.iloc[0]
            least_common = class_counts.iloc[-1]
            st.info(f"🌿 **Dominant Class:** `{most_common['Bean Variety']}` with {most_common['Count']:,} samples ({most_common['Percentage (%)']}%)")
            st.warning(f"🌾 **Smallest Class:** `{least_common['Bean Variety']}` with {least_common['Count']:,} samples ({least_common['Percentage (%)']}%)")
            
    with eda_sub3:
        st.markdown("#### Feature Distribution & Outlier Inspector")
        col_f1, col_f2 = st.columns([1, 2])
        
        with col_f1:
            selected_feature = st.selectbox("Select Feature to Inspect", feature_names, index=0)
            feat_series = df[selected_feature]
            
            st.metric("Mean", f"{feat_series.mean():.3f}")
            st.metric("Standard Deviation", f"{feat_series.std():.3f}")
            st.metric("Median (50%)", f"{feat_series.median():.3f}")
            st.metric("Skewness", f"{feat_series.skew():.3f}")
            st.metric("Kurtosis", f"{feat_series.kurtosis():.3f}")
            
        with col_f2:
            fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1.2]})
            
            # Histogram with KDE
            sns.histplot(feat_series, kde=True, ax=ax_hist, color="#2d6a4f", edgecolor="black", alpha=0.6)
            ax_hist.axvline(feat_series.mean(), color="red", linestyle="--", label=f"Mean: {feat_series.mean():.2f}")
            ax_hist.axvline(feat_series.median(), color="orange", linestyle=":", label=f"Median: {feat_series.median():.2f}")
            ax_hist.set_title(f"Distribution of '{selected_feature}'", fontsize=12, fontweight='bold')
            ax_hist.legend(fontsize=9)
            
            # Boxplot grouped by Class
            sns.boxplot(data=df, x="Class", y=selected_feature, ax=ax_box, palette="Set2")
            ax_box.set_title(f"'{selected_feature}' by Bean Variety", fontsize=11, fontweight='bold')
            ax_box.tick_params(axis='x', rotation=25)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
    with eda_sub4:
        st.markdown("#### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6.5))
        corr_matrix = X.corr()
        sns.heatmap(
            corr_matrix,
            cmap="vlag",
            annot=False,
            center=0,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Pearson Correlation Matrix (16 Numerical Features)", fontsize=13, fontweight='bold', pad=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("#### Bivariate Feature Relationship (Class Separation)")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            feat_x = st.selectbox("X-Axis Feature", feature_names, index=0)
        with col_sc2:
            feat_y = st.selectbox("Y-Axis Feature", feature_names, index=1)
            
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.scatterplot(
            data=df,
            x=feat_x,
            y=feat_y,
            hue="Class",
            alpha=0.65,
            palette="tab10",
            s=25,
            ax=ax
        )
        ax.set_title(f"{feat_x} vs. {feat_y} by Bean Variety", fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ==============================================================================
# TAB 2: SUPPORT VECTOR MACHINE (SVM)
# ==============================================================================
with tab_svm:
    st.markdown("""
    <div class="algo-header">
        <h3>⚡ Support Vector Machine (SVC)</h3>
        <p>SVM constructs hyperplanes in high-dimensional space to maximize the margin between classes. Features are standard scaled.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key="svm_k")
    with col_p2:
        svm_c = st.slider("Regularization (C)", min_value=0.01, max_value=20.0, value=1.0, step=0.1, key="svm_c")
    with col_p3:
        svm_gamma = st.selectbox("Gamma", ["scale", "auto"], index=0, key="svm_g")
        
    train_svm_btn = st.button("🚀 Train & Evaluate SVM", use_container_width=True, key="btn_svm")
    
    if train_svm_btn:
        with st.spinner("Training Support Vector Classifier..."):
            pipeline = make_pipeline(StandardScaler(), SVC(kernel=svm_kernel, C=svm_c, gamma=svm_gamma, probability=True, random_state=random_seed))
            evaluate_model("Support Vector Machine (SVM)", pipeline, X_train, X_test, y_train, y_test)
            st.success("✅ SVM Model trained and evaluated successfully!")
            
    if "Support Vector Machine (SVM)" in st.session_state["trained_models"]:
        res = st.session_state["trained_models"]["Support Vector Machine (SVM)"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{res['accuracy']*100:.2f}%")
        c2.metric("Weighted F1-Score", f"{res['f1']*100:.2f}%")
        c3.metric("Weighted Precision", f"{res['precision']*100:.2f}%")
        c4.metric("Training Time", f"{res['time']:.2f} s")
        
        st.markdown("---")
        col_cm, col_cr = st.columns([1.2, 1])
        with col_cm:
            st.pyplot(plot_confusion_matrix(res["cm"], class_names, "SVM Confusion Matrix"), use_container_width=True)
        with col_cr:
            st.markdown("##### Detailed Classification Report")
            rep_df = pd.DataFrame(res["report"]).T.round(3)
            st.dataframe(rep_df, use_container_width=True)
    else:
        st.info("Click 'Train & Evaluate SVM' above to run the algorithm.")

# ==============================================================================
# TAB 3: NAIVE BAYES
# ==============================================================================
with tab_nb:
    st.markdown("""
    <div class="algo-header">
        <h3>🧪 Gaussian Naive Bayes</h3>
        <p>Applies Bayes' theorem with the assumption of conditional feature independence and Gaussian normal distribution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        nb_var_smoothing = st.select_slider(
            "Variance Smoothing (Portion of largest variance added)",
            options=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            value=1e-9,
            key="nb_vs"
        )
    with col_p2:
        nb_scale = st.checkbox("Apply Standard Scaling", value=False, key="nb_scale")
        
    train_nb_btn = st.button("🚀 Train & Evaluate Naive Bayes", use_container_width=True, key="btn_nb")
    
    if train_nb_btn:
        with st.spinner("Training Gaussian Naive Bayes..."):
            if nb_scale:
                model = make_pipeline(StandardScaler(), GaussianNB(var_smoothing=nb_var_smoothing))
            else:
                model = GaussianNB(var_smoothing=nb_var_smoothing)
            evaluate_model("Naive Bayes", model, X_train, X_test, y_train, y_test)
            st.success("✅ Naive Bayes Model trained and evaluated successfully!")
            
    if "Naive Bayes" in st.session_state["trained_models"]:
        res = st.session_state["trained_models"]["Naive Bayes"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{res['accuracy']*100:.2f}%")
        c2.metric("Weighted F1-Score", f"{res['f1']*100:.2f}%")
        c3.metric("Weighted Precision", f"{res['precision']*100:.2f}%")
        c4.metric("Training Time", f"{res['time']:.2f} s")
        
        st.markdown("---")
        col_cm, col_cr = st.columns([1.2, 1])
        with col_cm:
            st.pyplot(plot_confusion_matrix(res["cm"], class_names, "Naive Bayes Confusion Matrix"), use_container_width=True)
        with col_cr:
            st.markdown("##### Detailed Classification Report")
            rep_df = pd.DataFrame(res["report"]).T.round(3)
            st.dataframe(rep_df, use_container_width=True)
    else:
        st.info("Click 'Train & Evaluate Naive Bayes' above to run the algorithm.")

# ==============================================================================
# TAB 4: DECISION TREE
# ==============================================================================
with tab_dt:
    st.markdown("""
    <div class="algo-header">
        <h3>🌳 Decision Tree Classifier</h3>
        <p>Non-parametric supervised learning algorithm that creates tree-based split rules based on feature thresholds.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        dt_criterion = st.selectbox("Split Criterion", ["gini", "entropy", "log_loss"], index=0, key="dt_crit")
    with col_p2:
        dt_max_depth = st.slider("Maximum Depth (0 = Unlimited)", min_value=0, max_value=25, value=10, step=1, key="dt_depth")
        max_d = None if dt_max_depth == 0 else dt_max_depth
    with col_p3:
        dt_min_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1, key="dt_split")
        
    train_dt_btn = st.button("🚀 Train & Evaluate Decision Tree", use_container_width=True, key="btn_dt")
    
    if train_dt_btn:
        with st.spinner("Training Decision Tree Classifier..."):
            model = DecisionTreeClassifier(
                criterion=dt_criterion,
                max_depth=max_d,
                min_samples_split=dt_min_split,
                random_state=random_seed
            )
            evaluate_model("Decision Tree", model, X_train, X_test, y_train, y_test)
            st.success("✅ Decision Tree Model trained and evaluated successfully!")
            
    if "Decision Tree" in st.session_state["trained_models"]:
        res = st.session_state["trained_models"]["Decision Tree"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{res['accuracy']*100:.2f}%")
        c2.metric("Weighted F1-Score", f"{res['f1']*100:.2f}%")
        c3.metric("Weighted Precision", f"{res['precision']*100:.2f}%")
        c4.metric("Training Time", f"{res['time']:.2f} s")
        
        st.markdown("---")
        col_cm, col_cr = st.columns([1.2, 1])
        with col_cm:
            st.pyplot(plot_confusion_matrix(res["cm"], class_names, "Decision Tree Confusion Matrix"), use_container_width=True)
        with col_cr:
            st.markdown("##### Detailed Classification Report")
            rep_df = pd.DataFrame(res["report"]).T.round(3)
            st.dataframe(rep_df, use_container_width=True)
            
        # Feature Importance Chart
        st.markdown("##### 🌟 Feature Importances")
        dt_model = res["model"]
        importances = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        importances.plot(kind="barh", color="#2d6a4f", ax=ax)
        ax.set_title("Decision Tree Feature Importance", fontsize=11, fontweight="bold")
        ax.set_xlabel("Gini / Entropy Importance")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Click 'Train & Evaluate Decision Tree' above to run the algorithm.")

# ==============================================================================
# TAB 5: LOGISTIC REGRESSION
# ==============================================================================
with tab_lr:
    st.markdown("""
    <div class="algo-header">
        <h3>📈 Logistic Regression</h3>
        <p>Linear model for multiclass classification using multinomial loss and soft-max probabilities. Features are standard scaled.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        lr_c = st.slider("Inverse Regularization Strength (C)", min_value=0.01, max_value=20.0, value=1.0, step=0.1, key="lr_c")
    with col_p2:
        lr_max_iter = st.slider("Max Iterations", min_value=200, max_value=3000, value=1000, step=200, key="lr_iter")
    with col_p3:
        lr_solver = st.selectbox("Solver", ["lbfgs", "saga", "newton-cg"], index=0, key="lr_solv")
        
    train_lr_btn = st.button("🚀 Train & Evaluate Logistic Regression", use_container_width=True, key="btn_lr")
    
    if train_lr_btn:
        with st.spinner("Training Logistic Regression Model..."):
            pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=lr_c, max_iter=lr_max_iter, solver=lr_solver, random_state=random_seed))
            evaluate_model("Logistic Regression", pipeline, X_train, X_test, y_train, y_test)
            st.success("✅ Logistic Regression Model trained and evaluated successfully!")
            
    if "Logistic Regression" in st.session_state["trained_models"]:
        res = st.session_state["trained_models"]["Logistic Regression"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{res['accuracy']*100:.2f}%")
        c2.metric("Weighted F1-Score", f"{res['f1']*100:.2f}%")
        c3.metric("Weighted Precision", f"{res['precision']*100:.2f}%")
        c4.metric("Training Time", f"{res['time']:.2f} s")
        
        st.markdown("---")
        col_cm, col_cr = st.columns([1.2, 1])
        with col_cm:
            st.pyplot(plot_confusion_matrix(res["cm"], class_names, "Logistic Regression Confusion Matrix"), use_container_width=True)
        with col_cr:
            st.markdown("##### Detailed Classification Report")
            rep_df = pd.DataFrame(res["report"]).T.round(3)
            st.dataframe(rep_df, use_container_width=True)
    else:
        st.info("Click 'Train & Evaluate Logistic Regression' above to run the algorithm.")

# ==============================================================================
# TAB 6: RANDOM FOREST
# ==============================================================================
with tab_rf:
    st.markdown("""
    <div class="algo-header">
        <h3>🌲 Random Forest Classifier</h3>
        <p>Ensemble of decision trees trained on bootstrap samples with random feature subsets to achieve lower variance and higher accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        rf_estimators = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=200, value=75, step=10, key="rf_est")
    with col_p2:
        rf_max_depth = st.slider("Max Depth (0 = Unlimited)", min_value=0, max_value=25, value=12, step=1, key="rf_depth")
        rf_d = None if rf_max_depth == 0 else rf_max_depth
    with col_p3:
        rf_criterion = st.selectbox("Criterion", ["gini", "entropy"], index=0, key="rf_crit")
        
    train_rf_btn = st.button("🚀 Train & Evaluate Random Forest", use_container_width=True, key="btn_rf")
    
    if train_rf_btn:
        with st.spinner("Training Random Forest Classifier..."):
            model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_d, criterion=rf_criterion, random_state=random_seed, n_jobs=1)
            evaluate_model("Random Forest", model, X_train, X_test, y_train, y_test)
            st.success("✅ Random Forest Model trained and evaluated successfully!")
            
    if "Random Forest" in st.session_state["trained_models"]:
        res = st.session_state["trained_models"]["Random Forest"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{res['accuracy']*100:.2f}%")
        c2.metric("Weighted F1-Score", f"{res['f1']*100:.2f}%")
        c3.metric("Weighted Precision", f"{res['precision']*100:.2f}%")
        c4.metric("Training Time", f"{res['time']:.2f} s")
        
        st.markdown("---")
        col_cm, col_cr = st.columns([1.2, 1])
        with col_cm:
            st.pyplot(plot_confusion_matrix(res["cm"], class_names, "Random Forest Confusion Matrix"), use_container_width=True)
        with col_cr:
            st.markdown("##### Detailed Classification Report")
            rep_df = pd.DataFrame(res["report"]).T.round(3)
            st.dataframe(rep_df, use_container_width=True)
            
        # Feature Importance Chart
        st.markdown("##### 🌟 Feature Importances")
        rf_model = res["model"]
        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        importances.plot(kind="barh", color="#1b4332", ax=ax)
        ax.set_title("Random Forest Mean Impurity Decrease", fontsize=11, fontweight="bold")
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Click 'Train & Evaluate Random Forest' above to run the algorithm.")

# ==============================================================================
# TAB 7: K-NEAREST NEIGHBORS (KNN)
# ==============================================================================
with tab_knn:
    st.markdown("""
    <div class="algo-header">
        <h3>🎯 K-Nearest Neighbors (KNN)</h3>
        <p>Instance-based classifier that assigns class labels based on majority vote among k closest neighbors in normalized space.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        knn_k = st.slider("Number of Neighbors (k)", min_value=1, max_value=30, value=7, step=1, key="knn_k")
    with col_p2:
        knn_weights = st.selectbox("Weight Function", ["uniform", "distance"], index=0, key="knn_w")
    with col_p3:
        knn_metric = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"], index=0, key="knn_m")
        
    train_knn_btn = st.button("🚀 Train & Evaluate KNN", use_container_width=True, key="btn_knn")
    
    if train_knn_btn:
        with st.spinner("Training K-Nearest Neighbors..."):
            pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights, metric=knn_metric))
            evaluate_model("KNN", pipeline, X_train, X_test, y_train, y_test)
            st.success("✅ KNN Model trained and evaluated successfully!")
            
    if "KNN" in st.session_state["trained_models"]:
        res = st.session_state["trained_models"]["KNN"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{res['accuracy']*100:.2f}%")
        c2.metric("Weighted F1-Score", f"{res['f1']*100:.2f}%")
        c3.metric("Weighted Precision", f"{res['precision']*100:.2f}%")
        c4.metric("Training Time", f"{res['time']:.2f} s")
        
        st.markdown("---")
        col_cm, col_cr = st.columns([1.2, 1])
        with col_cm:
            st.pyplot(plot_confusion_matrix(res["cm"], class_names, "KNN Confusion Matrix"), use_container_width=True)
        with col_cr:
            st.markdown("##### Detailed Classification Report")
            rep_df = pd.DataFrame(res["report"]).T.round(3)
            st.dataframe(rep_df, use_container_width=True)
    else:
        st.info("Click 'Train & Evaluate KNN' above to run the algorithm.")

# ==============================================================================
# TAB 8: MODEL COMPARISON & BENCHMARK
# ==============================================================================
with tab_compare:
    st.markdown("""
    <div class="algo-header">
        <h3>🏆 Multi-Algorithm Comparison & Benchmark</h3>
        <p>Benchmark all algorithms side-by-side on identical test splits to identify the optimal model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_bench1, col_bench2 = st.columns([1.5, 3])
    with col_bench1:
        run_all_btn = st.button("⚡ Train & Compare All 6 Algorithms", use_container_width=True)
        
    if run_all_btn:
        with st.spinner("Benchmarking all algorithms... This takes about 5 seconds."):
            models_to_run = {
                "Support Vector Machine (SVM)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=random_seed)),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
                "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_seed)),
                "Random Forest": RandomForestClassifier(n_estimators=75, random_state=random_seed, n_jobs=1),
                "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
            }
            for m_name, m_obj in models_to_run.items():
                evaluate_model(m_name, m_obj, X_train, X_test, y_train, y_test)
            st.success("🎉 All models benchmarked successfully!")
            
    if len(st.session_state["trained_models"]) > 0:
        summary_list = []
        for name, data in st.session_state["trained_models"].items():
            summary_list.append({
                "Algorithm": name,
                "Accuracy (%)": round(data["accuracy"] * 100, 2),
                "F1-Score (%)": round(data["f1"] * 100, 2),
                "Precision (%)": round(data["precision"] * 100, 2),
                "Recall (%)": round(data["recall"] * 100, 2),
                "Training Time (s)": round(data["time"], 3)
            })
            
        summary_df = pd.DataFrame(summary_list).sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
        
        best_row = summary_df.iloc[0]
        st.markdown(f"""
        <div style="background-color: #d8f3dc; border-left: 5px solid #2d6a4f; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="margin: 0; color: #1b4332;">🥇 Top Performer: <strong>{best_row['Algorithm']}</strong></h4>
            <p style="margin: 5px 0 0 0; color: #2d6a4f;">Achieved <strong>{best_row['Accuracy (%)']}%</strong> test accuracy and <strong>{best_row['F1-Score (%)']}%</strong> F1-score in {best_row['Training Time (s)']}s.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Performance Leaderboard")
        st.dataframe(summary_df, use_container_width=True)
        
        # Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = sns.barplot(
            data=summary_df,
            x="Algorithm",
            y="Accuracy (%)",
            palette="Greens_r",
            ax=ax
        )
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3),
                        textcoords='offset points')
        ax.set_ylim(min(summary_df["Accuracy (%)"].min() - 5, 60), 100)
        ax.set_title("Algorithm Test Accuracy Comparison (%)", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        plt.xticks(rotation=25, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No models trained yet. Click 'Train & Compare All 6 Algorithms' or train individual models in the tabs above!")

# ==============================================================================
# TAB 9: LIVE PREDICTION
# ==============================================================================
with tab_predict:
    st.markdown("""
    <div class="algo-header">
        <h3>🔮 Live Dry Bean Classifier</h3>
        <p>Input physical morphological measurements or load sample presets to predict the exact bean variety in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check available models
    available_model_names = list(st.session_state["trained_models"].keys())
    
    # Also check if saved dry_bean_model.joblib exists
    joblib_model_loaded = None
    if os.path.exists("dry_bean_model.joblib"):
        try:
            saved_j = joblib.load("dry_bean_model.joblib")
            joblib_model_loaded = saved_j.get("model")
            if "Saved Joblib Pipeline (Best Model)" not in available_model_names:
                available_model_names.append("Saved Joblib Pipeline (Best Model)")
        except Exception:
            pass
            
    if not available_model_names:
        st.warning("⚠️ No model is available for prediction yet. Please train a model in any of the ML tabs or benchmark tab first!")
    else:
        col_m_select, col_preset = st.columns([1.5, 1.5])
        with col_m_select:
            selected_model_name = st.selectbox("Select Model for Prediction", available_model_names, index=0)
            
        # Preset feature values from actual dataset for quick testing
        preset_beans = {
            "-- Choose a preset sample --": None,
            "SEKER (Round, compact)": df[df["Class"] == "SEKER"].iloc[0][feature_names].to_dict(),
            "BARBUNYA (Oval, speckled)": df[df["Class"] == "BARBUNYA"].iloc[0][feature_names].to_dict(),
            "BOMBAY (Very large bean)": df[df["Class"] == "BOMBAY"].iloc[0][feature_names].to_dict(),
            "CALI (Californian bean)": df[df["Class"] == "CALI"].iloc[0][feature_names].to_dict(),
            "DERMASON (Small standard)": df[df["Class"] == "DERMASON"].iloc[0][feature_names].to_dict(),
            "HOROZ (Long cylindrical)": df[df["Class"] == "HOROZ"].iloc[0][feature_names].to_dict(),
            "SIRA (Medium common)": df[df["Class"] == "SIRA"].iloc[0][feature_names].to_dict()
        }
        
        with col_preset:
            selected_preset = st.selectbox("Load Sample Bean Preset (Auto-fills inputs)", list(preset_beans.keys()), index=1)
            
        active_preset = preset_beans[selected_preset]
        
        st.markdown("#### Input Features (16 Morphological Measurements)")
        input_data = {}
        
        # Display inputs in 4 neat columns
        col_list = st.columns(4)
        for idx, feat in enumerate(feature_names):
            col_target = col_list[idx % 4]
            f_min = float(df[feat].min())
            f_max = float(df[feat].max())
            f_mean = float(df[feat].mean())
            
            val = float(active_preset[feat]) if active_preset is not None else f_mean
            # Ensure within min and max
            val = max(f_min, min(f_max, val))
            
            with col_target:
                input_data[feat] = st.number_input(
                    f"{feat}",
                    min_value=f_min,
                    max_value=f_max,
                    value=val,
                    format="%.4f" if f_max < 10 else "%.2f",
                    key=f"input_{feat}_{selected_preset}"
                )
                
        predict_btn = st.button("🌱 Predict Dry Bean Variety", use_container_width=True, type="primary")
        
        if predict_btn:
            input_df = pd.DataFrame([input_data])
            
            # Retrieve model
            if selected_model_name == "Saved Joblib Pipeline (Best Model)" and joblib_model_loaded is not None:
                predictor = joblib_model_loaded
            else:
                predictor = st.session_state["trained_models"][selected_model_name]["model"]
                
            pred_numeric = predictor.predict(input_df)[0]
            if isinstance(pred_numeric, (int, np.integer)):
                pred_label = label_encoder.inverse_transform([pred_numeric])[0]
            else:
                pred_label = str(pred_numeric)
                
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-title">Prediction Result</div>
                <div class="prediction-result">🌱 {pred_label}</div>
                <div style="color: #2d6a4f; font-weight: 500;">Classified by {selected_model_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities if available
            if hasattr(predictor, "predict_proba"):
                try:
                    probs = predictor.predict_proba(input_df)[0]
                    prob_df = pd.DataFrame({
                        "Bean Variety": class_names,
                        "Confidence (%)": (probs * 100).round(2)
                    }).sort_values(by="Confidence (%)", ascending=False)
                    
                    st.markdown("##### Prediction Confidence Probabilities")
                    col_p1, col_p2 = st.columns([1, 1.5])
                    with col_p1:
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    with col_p2:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        sns.barplot(
                            data=prob_df,
                            x="Confidence (%)",
                            y="Bean Variety",
                            palette="Greens_r",
                            ax=ax
                        )
                        ax.set_title("Class Probabilities", fontsize=11, fontweight="bold")
                        ax.set_xlim(0, 100)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                except Exception:
                    pass
