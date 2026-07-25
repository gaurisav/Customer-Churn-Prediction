import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_resource
def load_artifacts():
    mp = os.path.join(MODELS_DIR, "churn_model.pkl")
    sp = os.path.join(MODELS_DIR, "scaler.pkl")
    return joblib.load(mp) if os.path.exists(mp) else None, joblib.load(sp) if os.path.exists(sp) else None

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

@st.cache_data
def load_processed_data():
    if not os.path.exists(DATA_PATH):
        return None, None
    df = load_dataset().copy()
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    df = df.drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    bm = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for c in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[c] = df[c].map(bm)
    mm = {
        "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
        "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
        "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
        "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
        "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaymentMethod": {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
    }
    for c, m in mm.items():
        df[c] = df[c].map(m)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop("Churn", axis=1)
    return X, y

model, scaler = load_artifacts()
df_raw = load_dataset()
X_full, y_full = load_processed_data()

if df_raw is not None:
    total_customers = len(df_raw)
    churn_count = df_raw["Churn"].value_counts().get("Yes", 0)
    churn_rate = churn_count / total_customers * 100
    avg_tenure = df_raw["tenure"].mean()
    avg_monthly = df_raw["MonthlyCharges"].mean()

def preprocess_single_input(input_dict):
    df = pd.DataFrame([input_dict])
    bm = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    for c in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if c in df.columns:
            df[c] = df[c].map(bm)
    mm = {
        "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
        "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
        "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
        "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
        "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaymentMethod": {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
    }
    for c, m in mm.items():
        if c in df.columns:
            df[c] = df[c].map(m)
    df["Tenure_Group"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 48, 100], labels=[0, 1, 2, 3, 4], right=True).astype(int)
    df["MonthlySpendingCategory"] = pd.cut(df["MonthlyCharges"], bins=[0, 40, 80, 150], labels=[0, 1, 2], right=True).astype(int)
    df["LongTermCustomer"] = (df["tenure"] >= 48).astype(int)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    sc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["ServiceCount"] = 0
    for c in sc:
        df["ServiceCount"] += (df[c] == 1).astype(int)
    df["HasPartnerAndDependents"] = ((df["Partner"] == 1) & (df["Dependents"] == 1)).astype(int)
    df["IsSeniorWithPartner"] = ((df["SeniorCitizen"] == 1) & (df["Partner"] == 1)).astype(int)
    df["TenureXMonthlyCharges"] = df["tenure"] * df["MonthlyCharges"]
    if scaler is not None:
        for c in ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "ServiceCount", "TenureXMonthlyCharges"]:
            if c in df.columns:
                df[c] = scaler.transform(df[[c]].values.astype(float))
    return df

def get_recommendations(inp, prob):
    recs = []
    if inp["Contract"] == "Month-to-month":
        recs.append({"problem": "Month-to-month contract", "action": "Promote annual contract with 1-month free discount", "icon": "📄", "impact": "Can reduce churn risk by 40%"})
    if inp["TechSupport"] == "No":
        recs.append({"problem": "No Tech Support", "action": "Offer free 30-day Tech Support trial", "icon": "🛠️", "impact": "2x more likely to stay with support"})
    if inp["OnlineSecurity"] == "No":
        recs.append({"problem": "No Online Security", "action": "Bundle Online Security free for 3 months", "icon": "🔒", "impact": "3x lower churn with security"})
    if inp["InternetService"] == "Fiber optic" and inp["MonthlyCharges"] > 80:
        recs.append({"problem": "Fiber optic + high charges", "action": "Speed upgrade + price lock 12 months", "icon": "🌐", "impact": "Fiber churn at 42%"})
    if inp["tenure"] < 12:
        recs.append({"problem": "New customer (< 12mo)", "action": "Loyalty program: 10% off first 6 months", "icon": "🌟", "impact": "Reduces churn 25%"})
    if inp["PaymentMethod"] == "Electronic check":
        recs.append({"problem": "Electronic check", "action": "Auto-pay incentive: $5/month off", "icon": "💳", "impact": "Retention +15%"})
    if inp["MonthlyCharges"] > 80:
        recs.append({"problem": "High charges >$80", "action": "15% loyalty discount for 3 months", "icon": "💰", "impact": "Reduces risk 20%"})
    if inp.get("Dependents") == "No" and inp.get("Partner") == "No":
        recs.append({"problem": "Single no dependents", "action": "Family bundle discount", "icon": "👤", "impact": "Family plans 92% retention"})
    urgency = "🔴 HIGH RISK - Immediate action needed" if prob > 0.7 else ("🟡 MEDIUM RISK - Proactive outreach recommended" if prob > 0.4 else "🟢 LOW RISK - Monitor and maintain")
    return recs, urgency

st.sidebar.title("📊 Customer Churn")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📋 Project Overview", "📊 Dataset Summary", "🎯 KPIs", "📈 Data Analysis", "🔮 Prediction"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:** Streamlit · scikit-learn · SHAP")
st.sidebar.markdown("**Model:** Logistic Regression (F1: 0.565, AUC: 0.84)")

# ==================== HOME ====================
if page == "🏠 Home":
    st.title("🏠 Customer Churn Prediction Dashboard")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Welcome!
        This interactive dashboard helps you **understand, predict, and prevent** customer churn.
        ### What you can do:
        - **Explore** the dataset and key metrics
        - **Visualize** churn patterns across segments
        - **Predict** churn risk for individual customers
        - **Understand** why via SHAP explainability
        - **Get recommendations** on how to retain them
        ### Model Performance
        | Metric | Value |
        |--------|-------|
        | **Accuracy** | 79.9% |
        | **ROC-AUC** | 0.84 |
        | **Precision** | 66.0% |
        | **Recall** | 49.5% |
        ### Business Impact
        - Annual churn cost: ~$1.45M | Potential savings: ~$466K/yr | Customers saved: ~600/yr
        """)
    with col2:
        if df_raw is not None:
            st.metric("Total Customers", f"{total_customers:,}")
            st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-26.5% annual")
            st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
            st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
            st.markdown("---")
            st.markdown("### Churn Distribution")
            fig, ax = plt.subplots(figsize=(4, 4))
            c = df_raw["Churn"].value_counts()
            ax.pie(c.values, labels=["No Churn", "Churn"], autopct="%1.1f%%", colors=["#4CAF50","#FF5722"], startangle=90, explode=(0.03, 0.03))
            st.pyplot(fig)
            plt.close()
    st.success("👈 Use sidebar to navigate. Start with **Dataset Summary**!")

# ==================== PROJECT OVERVIEW ====================
elif page == "📋 Project Overview":
    st.title("📋 Project Overview")
    st.markdown("---")
    st.markdown("""
    ## Problem Statement
    A telecom company loses **~26.5% of customers** annually costing **$1.45M**. This tool identifies at-risk customers early.
    ## Dataset
    | Detail | Value |
    |--------|-------|
    | **Source** | IBM Telco Customer Churn Dataset |
    | **Samples** | 7,043 customers |
    | **Features** | 21 (demographics, services, billing) |
    | **Target** | `Churn` (Yes / No) |
    | **Class Balance** | 73.5% No Churn · 26.5% Churn |
    ## Methodology
    **Preprocessing:** Converted TotalCharges, removed 22 duplicates, encoded 15 categorical features, scaled numerics.
    **Feature Engineering (8 new features):** Tenure_Group, MonthlySpendingCategory, LongTermCustomer, AvgMonthlySpend, ServiceCount, HasPartnerAndDependents, IsSeniorWithPartner, TenureXMonthlyCharges
    ## Models Compared
    | Model | F1 | ROC-AUC |
    |-------|----|---------|
    | Logistic Regression | **0.565** | **0.841** |
    | Random Forest (tuned) | 0.562 | 0.840 |
    | Random Forest (baseline) | 0.550 | 0.835 |
    | Decision Tree | 0.528 | 0.785 |
    ## Top Churn Drivers (SHAP)
    1. **Contract** (Month-to-month) | 2. **Tenure** (Low) | 3. **Monthly Charges** (High) | 4. **Service Count** (Few) | 5. **Internet Service** (Fiber optic)
    """)

# ==================== DATASET SUMMARY ====================
elif page == "📊 Dataset Summary":
    st.title("📊 Dataset Summary")
    st.markdown("---")
    if df_raw is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistical Summary")
            st.dataframe(df_raw.describe(), use_container_width=True)
        with col2:
            st.subheader("Column Info")
            buf = pd.DataFrame({
                "Column": df_raw.dtypes.index,
                "Type": df_raw.dtypes.values,
                "Non-Null": df_raw.count().values,
                "Null": df_raw.isnull().sum().values,
                "Unique": [df_raw[c].nunique() for c in df_raw.columns]
            })
            st.dataframe(buf, use_container_width=True)
        st.subheader("Sample Data (First 10 Rows)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.subheader("Churn Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_raw["Churn"].value_counts().reset_index().rename(columns={"index":"Churn","Churn":"Count"}), use_container_width=True)
        with col2:
            st.dataframe((df_raw["Churn"].value_counts(normalize=True)*100).reset_index().rename(columns={"index":"Churn","Churn":"%"}), use_container_width=True)

# ==================== KPIs ====================
elif page == "🎯 KPIs":
    st.title("🎯 Key Performance Indicators")
    st.markdown("---")
    if df_raw is not None:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Total Customers", f"{total_customers:,}")
        with k2: st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-26.5% annual")
        with k3: st.metric("Avg Tenure", f"{avg_tenure:.1f}mo")
        with k4: st.metric("Avg Monthly", f"${avg_monthly:.2f}")
        with k5: st.metric("Churners", f"{churn_count:,}")
        st.markdown("---")
        st.subheader("Segment KPIs")
        col1, col2 = st.columns(2)
        with col1:
            for t, c in [("Contract","Contract"),("Internet Service","InternetService"),("Tech Support","TechSupport")]:
                d = df_raw.groupby(c)["Churn"].apply(lambda x: (x=="Yes").mean()*100)
                st.markdown(f"**Churn Rate by {t}**")
                st.dataframe(d.reset_index().rename(columns={"Churn":"%"}).round(2), use_container_width=True)
        with col2:
            for t, c in [("Payment","PaymentMethod"),("Senior Citizen","SeniorCitizen"),("Partner","Partner")]:
                d = df_raw.groupby(c)["Churn"].apply(lambda x: (x=="Yes").mean()*100)
                st.markdown(f"**Churn Rate by {t}**")
                st.dataframe(d.reset_index().rename(columns={"Churn":"%"}).round(2), use_container_width=True)
        st.subheader("Financial Impact")
        c1, c2, c3 = st.columns(3)
        loss = churn_count * avg_monthly * 12
        with c1: st.metric("Annual Revenue Lost", f"${loss:,.0f}")
        with c2: st.metric("Potential Savings", f"${loss*0.322:,.0f}")
        with c3: st.metric("Retainable/yr", f"{int(churn_count*0.322):,}")

# ==================== DATA ANALYSIS ====================
elif page == "📈 Data Analysis":
    st.title("📈 Data Analysis")
    st.markdown("---")
    if df_raw is not None:
        t1, t2, t3, t4 = st.tabs(["📄 Contract", "💰 Monthly Charges", "⏱ Tenure", "📊 Churn Analysis"])
        with t1:
            st.subheader("Churn by Contract Type")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8,5))
                ax.pie(df_raw["Contract"].value_counts().values, labels=df_raw["Contract"].value_counts().index, autopct="%1.1f%%", colors=["#FF9999","#66B2FF","#99FF99"], startangle=90)
                ax.set_title("Contract Distribution", fontweight="bold")
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(8,5))
                pd.crosstab(df_raw["Contract"], df_raw["Churn"], normalize="index")*100 \
                    .plot(kind="bar", ax=ax, color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Churn Rate by Contract"); ax.set_ylabel("%"); ax.legend(); ax.tick_params(axis="x",rotation=0)
                st.pyplot(fig); plt.close()
            st.info("💡 Month-to-month churn at **42.7%** vs Two-year at **2.9%**.")
        with t2:
            st.subheader("Churn by Monthly Charges")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8,5))
                ax.hist([df_raw[df_raw["Churn"]=="No"]["MonthlyCharges"], df_raw[df_raw["Churn"]=="Yes"]["MonthlyCharges"]], bins=30, alpha=0.7, label=["No Churn","Churn"], color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Monthly Charges Distribution"); ax.legend()
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(8,5))
                sns.boxplot(x="Churn", y="MonthlyCharges", data=df_raw, ax=ax, palette=["#4CAF50","#FF5722"])
                ax.set_title("Charges by Churn")
                st.pyplot(fig); plt.close()
            st.info("💡 Churners avg $74/mo vs loyal $61/mo.")
        with t3:
            st.subheader("Churn by Tenure")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8,5))
                ax.hist([df_raw[df_raw["Churn"]=="No"]["tenure"], df_raw[df_raw["Churn"]=="Yes"]["tenure"]], bins=30, alpha=0.7, label=["No Churn","Churn"], color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Tenure Distribution"); ax.legend()
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(8,5))
                sns.boxplot(x="Churn", y="tenure", data=df_raw, ax=ax, palette=["#4CAF50","#FF5722"])
                ax.set_title("Tenure by Churn")
                st.pyplot(fig); plt.close()
            st.info("💡 Churners avg 18 months vs loyal 38 months.")
        with t4:
            st.subheader("Comprehensive Churn Analysis")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8,5))
                pd.crosstab(df_raw["InternetService"], df_raw["Churn"], normalize="index")*100 \
                    .plot(kind="bar", ax=ax, color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Churn by Internet"); ax.legend(); ax.tick_params(axis="x",rotation=0)
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(8,5))
                pd.crosstab(df_raw["TechSupport"], df_raw["Churn"], normalize="index")*100 \
                    .plot(kind="bar", ax=ax, color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Churn by Tech Support"); ax.legend(); ax.tick_params(axis="x",rotation=0)
                st.pyplot(fig); plt.close()
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8,5))
                pd.crosstab(df_raw["PaymentMethod"], df_raw["Churn"], normalize="index")*100 \
                    .plot(kind="bar", ax=ax, color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Churn by Payment"); ax.legend(); ax.tick_params(axis="x",rotation=15)
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(8,5))
                pd.crosstab(df_raw["OnlineSecurity"], df_raw["Churn"], normalize="index")*100 \
                    .plot(kind="bar", ax=ax, color=["#4CAF50","#FF5722"], edgecolor="black")
                ax.set_title("Churn by Online Security"); ax.legend(); ax.tick_params(axis="x",rotation=0)
                st.pyplot(fig); plt.close()

# ==================== PREDICTION ====================
elif page == "🔮 Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    if model is None:
        st.warning("⚠️ No trained model found. Run `python src/train_model.py` first.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📝 Enter Customer Details")
            with st.form("pred_form"):
                gender = st.selectbox("Gender", ["Male","Female"])
                senior = st.selectbox("Senior Citizen", [0,1])
                partner = st.selectbox("Partner", ["Yes","No"])
                dependents = st.selectbox("Dependents", ["Yes","No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                phone = st.selectbox("Phone Service", ["Yes","No"])
                multi = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
                internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
                onlinesec = st.selectbox("Online Security", ["No","Yes","No internet service"])
                onlinebackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
                device = st.selectbox("Device Protection", ["No","Yes","No internet service"])
                tech = st.selectbox("Tech Support", ["No","Yes","No internet service"])
                stv = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
                smov = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes","No"])
                payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
                monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
                total = st.slider("Total Charges ($)", 18.0, 8700.0, 1500.0)
                submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

            if submitted:
                inp = {
                    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
                    "MultipleLines": multi, "InternetService": internet,
                    "OnlineSecurity": onlinesec, "OnlineBackup": onlinebackup,
                    "DeviceProtection": device, "TechSupport": tech,
                    "StreamingTV": stv, "StreamingMovies": smov,
                    "Contract": contract, "PaperlessBilling": paperless,
                    "PaymentMethod": payment, "MonthlyCharges": monthly,
                    "TotalCharges": total
                }
                proc = preprocess_single_input(inp)
                prob = float(model.predict_proba(proc)[0][1])
                pred = int(model.predict(proc)[0])
                st.session_state["prediction"] = pred
                st.session_state["probability"] = prob
                st.session_state["input_dict"] = inp
                st.session_state["processed"] = proc

        with col2:
            if "prediction" in st.session_state:
                pred = st.session_state["prediction"]
                prob = st.session_state["probability"]
                inp = st.session_state["input_dict"]

                # PREDICTION RESULT
                st.subheader("📊 Prediction Result")
                if pred == 1:
                    st.error(f"### ⚠️ High Risk - Likely to CHURN")
                else:
                    st.success(f"### ✅ Low Risk - Likely to STAY")
                st.markdown(f"### Probability = {prob:.1%}")
                st.progress(float(prob))
                st.markdown("---")
                risk = "🔴 HIGH" if prob > 0.7 else ("🟡 MEDIUM" if prob > 0.4 else "🟢 LOW")
                st.metric("Risk Level", risk, delta=f"{prob:.1%} probability")

                # SHAP EXPLANATION
                st.markdown("---")
                st.subheader("🔍 Explainability - Why this prediction?")
                try:
                    if X_full is not None:
                        X_sample = X_full.sample(n=min(100, len(X_full)), random_state=42)
                        mtype = type(model).__name__
                        if mtype == "LogisticRegression":
                            explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
                        elif mtype in ["RandomForestClassifier","GradientBoostingClassifier"]:
                            explainer = shap.TreeExplainer(model)
                        else:
                            explainer = shap.KernelExplainer(model.predict_proba, X_sample.sample(n=min(50,len(X_sample)),random_state=42))
                        shap_vals = explainer.shap_values(X_sample)
                        shap_churn = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                        proc = st.session_state["processed"]
                        proc_aligned = proc.reindex(columns=X_sample.columns, fill_value=0)
                        shap_single = explainer.shap_values(proc_aligned)
                        shap_single = shap_single[1] if isinstance(shap_single, list) else shap_single
                        if len(np.array(shap_single).shape) == 2 and np.array(shap_single).shape[0] == 1:
                            shap_single = shap_single[0]
                        feat_names = X_sample.columns.tolist()
                        shap_df = pd.DataFrame({"feature": feat_names, "shap_value": shap_single})
                        shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
                        shap_df = shap_df.sort_values("abs_shap", ascending=False)
                        top_n = min(10, len(shap_df))
                        top_feat = shap_df.head(top_n).iloc[::-1]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ["#FF5722" if v > 0 else "#4CAF50" for v in top_feat["shap_value"]]
                        ax.barh(range(len(top_feat)), top_feat["shap_value"].values, color=colors, edgecolor="black")
                        ax.set_yticks(range(len(top_feat)))
                        ax.set_yticklabels(top_feat["feature"].values, fontsize=10)
                        ax.axvline(x=0, color="black", linewidth=0.5)
                        ax.set_xlabel("SHAP Value (impact on model output)")
                        ax.set_title("Top Factors Driving This Prediction", fontweight="bold")
                        for i, (_, r) in enumerate(top_feat.iterrows()):
                            ax.text(r["shap_value"] + (0.01 if r["shap_value"]>0 else -0.01), i, f"{r['shap_value']:+.4f}", va="center", fontsize=9, fontweight="bold")
                        st.pyplot(fig)
                        plt.close()
                        st.markdown("**Top Contributing Factors:**")
                        for i, (_, r) in enumerate(shap_df.head(5).iterrows()):
                            d = "⬆ Increases churn risk" if r["shap_value"]>0 else "⬇ Decreases churn risk"
                            st.markdown(f"{i+1}. **{r['feature']}** {d} (SHAP: {r['shap_value']:+.4f})")
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable, showing model coefficients instead.")
                    if hasattr(model, "coef_"):
                        coefs = model.coef_[0]
                        if X_full is not None:
                            fi = pd.DataFrame({"feature": X_full.columns[:len(coefs)], "coef": coefs})
                            fi["abs"] = np.abs(fi["coef"])
                            fi = fi.sort_values("abs", ascending=False).head(10)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            fsorted = fi.sort_values("coef")
                            colors = ["#FF5722" if v > 0 else "#4CAF50Let me check the current state of the app.py file to see if it's complete enough to run.

<read_file>
<path>C:\Users\gauri\OneDrive\Desktop\customer_churn\Customer-Churn-Prediction\app.py</path>
</read_file>
