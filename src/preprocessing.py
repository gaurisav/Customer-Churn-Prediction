import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Load the Telco customer churn dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file (default: "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """
    Complete preprocessing pipeline for training.

    Steps performed:
        1. Drop customerID column
        2. Remove duplicate rows
        3. Handle TotalCharges datatype (convert to numeric)
        4. Handle missing values (fill with 0 or mode)
        5. Encode categorical variables
        6. Feature engineering (create new useful features)
        7. Split into X (features) and y (target)
        8. Scale numerical features

    New features created:
        - Tenure_Group           : Binned tenure into categories (0-6, 7-12, 13-24, 25-48, 49+ months)
        - MonthlySpendingCategory: Binned MonthlyCharges (Low/Medium/High)
        - LongTermCustomer       : Flag (1 if tenure >= 48 months)
        - AvgMonthlySpend        : TotalCharges / (tenure + 1) — average spend per month
        - ServiceCount           : Count of subscribed services (out of 6)
        - HasPartnerAndDependents: 1 if both Partner and Dependents are Yes
        - IsSeniorWithPartner    : 1 if SeniorCitizen and has Partner
        - TenureXMonthlyCharges  : tenure * MonthlyCharges interaction feature

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (1=Churn, 0=No Churn)
    label_encoders : dict
        Dictionary of fitted LabelEncoders for each categorical column
    scaler : StandardScaler
        Fitted StandardScaler for numerical columns
    """
    original_shape = df.shape

    # ------------------------------------------------------------------
    # 1. Drop customerID column
    # ------------------------------------------------------------------
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
        print("Dropped 'customerID' column.")

    # ------------------------------------------------------------------
    # 2. Remove duplicates
    # ------------------------------------------------------------------
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {dup_count} duplicate row(s).")
    else:
        print("No duplicate rows found.")

    # ------------------------------------------------------------------
    # 3. Handle TotalCharges datatype
    # ------------------------------------------------------------------
    # Some values may be empty strings; convert to numeric forcing errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print("Converted 'TotalCharges' to numeric datatype.")

    # ------------------------------------------------------------------
    # 4. Handle missing values
    # ------------------------------------------------------------------
    # Check for any nulls
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"Missing values found:\n{null_cols}")
        # Fill TotalCharges NaN with 0 (new customers with no charges)
        if "TotalCharges" in null_cols.index:
            df["TotalCharges"] = df["TotalCharges"].fillna(0)
            print("Filled NaN in 'TotalCharges' with 0.")
        # For any other NaN in categorical columns, fill with mode
        for col in null_cols.index:
            if col != "TotalCharges":
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled NaN in '{col}' with mode value '{mode_val}'.")
    else:
        print("No missing values found.")

    # ------------------------------------------------------------------
    # 5. Encode categorical variables
    # ------------------------------------------------------------------
    label_encoders = {}

    # 5a. Binary categorical columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Encoded binary categorical columns:", binary_cols)

    # 5b. Multi-class categorical columns
    multi_class_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    for col in multi_class_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Encoded multi-class categorical columns:", multi_class_cols)

    # ------------------------------------------------------------------
    # 6. Feature Engineering — Create new useful features
    # ------------------------------------------------------------------
    print("\n--- Feature Engineering ---")

    # 6a. Tenure Group — bin tenure into meaningful groups
    tenure_bins = [0, 6, 12, 24, 48, 100]
    tenure_labels = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]
    df["Tenure_Group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True)
    le_tenure = LabelEncoder()
    df["Tenure_Group"] = le_tenure.fit_transform(df["Tenure_Group"])
    label_encoders["Tenure_Group"] = le_tenure
    print("Created 'Tenure_Group' — binned tenure into 5 categories.")

    # 6b. Monthly Spending Category
    spend_bins = [0, 40, 80, 150]
    spend_labels = ["Low", "Medium", "High"]
    df["MonthlySpendingCategory"] = pd.cut(df["MonthlyCharges"], bins=spend_bins, labels=spend_labels, right=True)
    le_spend = LabelEncoder()
    df["MonthlySpendingCategory"] = le_spend.fit_transform(df["MonthlySpendingCategory"])
    label_encoders["MonthlySpendingCategory"] = le_spend
    print("Created 'MonthlySpendingCategory' — Low (<$40), Medium ($40-$80), High (>$80).")

    # 6c. Long Term Customer — tenure >= 48 months (2+ years)
    df["LongTermCustomer"] = (df["tenure"] >= 48).astype(int)
    print("Created 'LongTermCustomer' — 1 if tenure >= 48 months.")

    # 6d. Average Monthly Spend
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    print("Created 'AvgMonthlySpend' — TotalCharges / (tenure + 1).")

    # 6e. Service Count — count of subscribed add-on services (value 1 = Yes after encoding)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    df["ServiceCount"] = 0
    for col in service_cols:
        df["ServiceCount"] += (df[col] == 1).astype(int)
    print("Created 'ServiceCount' — count of subscribed services.")

    # 6f. Has Partner and Dependents (interaction feature)
    df["HasPartnerAndDependents"] = ((df["Partner"] == 1) & (df["Dependents"] == 1)).astype(int)
    print("Created 'HasPartnerAndDependents' — 1 if both Partner and Dependents are Yes.")

    # 6g. Senior with Partner
    df["IsSeniorWithPartner"] = ((df["SeniorCitizen"] == 1) & (df["Partner"] == 1)).astype(int)
    print("Created 'IsSeniorWithPartner' — 1 if SeniorCitizen and has Partner.")

    # 6h. Interaction: Tenure x MonthlyCharges (multiplicative)
    df["TenureXMonthlyCharges"] = df["tenure"] * df["MonthlyCharges"]
    print("Created 'TenureXMonthlyCharges' — tenure * MonthlyCharges interaction.")

    print("Feature engineering complete. Added 8 new features.")

    # ------------------------------------------------------------------
    # 7. Split X and y
    # ------------------------------------------------------------------
    if "Churn" in df.columns:
        X = df.drop("Churn", axis=1)
        y = df["Churn"].map({"Yes": 1, "No": 0})
        print("\nSplit into X (features) and y (target).")
    else:
        X = df
        y = None
        print("\nNo 'Churn' column found; returning X only.")

    # ------------------------------------------------------------------
    # 8. Scale numerical features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "AvgMonthlySpend", "ServiceCount", "TenureXMonthlyCharges"]
    # Only scale columns that exist
    num_cols = [c for c in num_cols if c in X.columns]
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print("Scaled numerical features:", num_cols)

    # Final summary
    print(f"\nPreprocessing complete: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, label_encoders, scaler


def preprocess_input(df):
    """
    Preprocess a single customer record for prediction (inference-time).

    Parameters
    ----------
    df : pd.DataFrame
        Single-row DataFrame with raw input values

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for model prediction
    """
    # Encode binary categorical variables
    binary_mapping = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0}
    }
    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Encode multi-class categorical variables
    multi_class_mapping = {
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
    for col, mapping in multi_class_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Testing preprocessing pipeline")
    print("=" * 60)
    df = load_data()
    X, y, encoders, scaler = preprocess_data(df)
    print(f"\nX shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
    print(f"\nNew feature columns: {[c for c in X.columns if c not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'SeniorCitizen']]}")
