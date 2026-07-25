# Generate key visualization graphs for Customer Churn Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(data_path)

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# ============================================================
# 1. Churn Distribution
# ============================================================
def plot_churn_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    churn_counts = df['Churn'].value_counts()
    colors = ['#4CAF50', '#FF5722']
    
    # Bar plot
    axes[0].bar(['No Churn', 'Churn'], churn_counts.values, color=colors, edgecolor='black', width=0.5)
    axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Customers')
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=12)
    
    # Pie chart
    axes[1].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0.05))
    axes[1].set_title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'churn_distribution.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 2. Gender Distribution
# ============================================================
def plot_gender_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    gender_counts = df['gender'].value_counts()
    axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                colors=['#66B2FF', '#FF9999'], startangle=90)
    axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    
    gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index') * 100
    gender_churn.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[1].set_title('Churn Rate by Gender', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xlabel('Gender')
    axes[1].legend(title='Churn')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'gender_distribution.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 3. Senior Citizen
# ============================================================
def plot_senior_citizen():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    senior_counts = df['SeniorCitizen'].value_counts().sort_index()
    labels = ['Non-Senior (0)', 'Senior (1)']
    axes[0].pie(senior_counts.values, labels=labels, autopct='%1.1f%%',
                colors=['#66B2FF', '#FF9999'], startangle=90)
    axes[0].set_title('Senior Citizen Distribution', fontsize=14, fontweight='bold')
    
    senior_churn = pd.crosstab(df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'}), df['Churn'], normalize='index') * 100
    senior_churn.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[1].set_title('Churn Rate by Senior Citizen Status', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(title='Churn')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'senior_citizen.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 4. Contract Type
# ============================================================
def plot_contract_type():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    contract_counts = df['Contract'].value_counts()
    axes[0].pie(contract_counts.values, labels=contract_counts.index, autopct='%1.1f%%',
                colors=['#FF9999', '#66B2FF', '#99FF99'], startangle=90)
    axes[0].set_title('Contract Type Distribution', fontsize=14, fontweight='bold')
    
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[1].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Churn')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'contract_type.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 5. Internet Service
# ============================================================
def plot_internet_service():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    inet_counts = df['InternetService'].value_counts()
    axes[0].pie(inet_counts.values, labels=inet_counts.index, autopct='%1.1f%%',
                colors=['#66B2FF', '#FF9999', '#99FF99'], startangle=90)
    axes[0].set_title('Internet Service Distribution', fontsize=14, fontweight='bold')
    
    inet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    inet_churn.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[1].set_title('Churn Rate by Internet Service', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Churn')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'internet_service.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 6. Monthly Charges
# ============================================================
def plot_monthly_charges():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
    not_churned = df[df['Churn'] == 'No']['MonthlyCharges']
    
    axes[0].hist([not_churned, churned], bins=30, alpha=0.7,
                 label=['No Churn', 'Churn'], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[0].set_title('Monthly Charges Distribution by Churn', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Monthly Charges ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1], palette=['#4CAF50', '#FF5722'])
    axes[1].set_title('Monthly Charges by Churn (Box Plot)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Churn')
    axes[1].set_ylabel('Monthly Charges ($)')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'monthly_charges.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 7. Tenure
# ============================================================
def plot_tenure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    churned = df[df['Churn'] == 'Yes']['tenure']
    not_churned = df[df['Churn'] == 'No']['tenure']
    
    axes[0].hist([not_churned, churned], bins=30, alpha=0.7,
                 label=['No Churn', 'Churn'], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[0].set_title('Tenure Distribution by Churn', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tenure (months)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[1], palette=['#4CAF50', '#FF5722'])
    axes[1].set_title('Tenure by Churn (Box Plot)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Churn')
    axes[1].set_ylabel('Tenure (months)')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tenure.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 8. Payment Method
# ============================================================
def plot_payment_method():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    payment_counts = df['PaymentMethod'].value_counts()
    axes[0] = plt.subplot(1, 2, 1)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(payment_counts)))
    axes[0].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
    axes[0].set_title('Payment Method Distribution', fontsize=14, fontweight='bold')
    
    axes[1] = plt.subplot(1, 2, 2)
    payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
    payment_churn.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FF5722'], edgecolor='black')
    axes[1].set_title('Churn Rate by Payment Method', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend(title='Churn')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'payment_method.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# 9. Correlation Heatmap
# ============================================================
def plot_correlation_heatmap():
    df_encoded = df.copy()
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop('customerID', axis=1)
    
    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Encode multi-class categorical columns
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod']
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
    
    plt.figure(figsize=(16, 12))
    correlation_matrix = df_encoded.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of All Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ============================================================
# Generate all graphs
# ============================================================
if __name__ == '__main__':
    print('Generating graphs...\n')
    plot_churn_distribution()
    plot_gender_distribution()
    plot_senior_citizen()
    plot_contract_type()
    plot_internet_service()
    plot_monthly_charges()
    plot_tenure()
    plot_payment_method()
    plot_correlation_heatmap()
    print('\n✅ All 9 graphs generated successfully in the images/ folder!')

