# Customer Churn Prediction

A machine learning project to predict customer churn for a telecommunications company. Built with Python, Scikit-learn, and Streamlit.

## Project Structure

```
Customer-Churn-Prediction
│
├── data/                  # Raw dataset (Telco Customer Churn)
├── notebooks/             # EDA notebook
├── src/                   # Source code (preprocessing, training, prediction)
├── models/                # Trained model artifacts
├── images/                # Visualizations & analysis plots
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gaurisav/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   ```
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

## Exploratory Data Analysis

Key insights from the dataset:

| Insight | Visualization |
|---------|--------------|
| **Churn Distribution** — ~26.5% of customers churn | ![Churn Distribution](images/churn_distribution.png) |
| **Tenure vs Churn** — Churned customers have lower tenure (~18 months avg) | ![Tenure Distribution](images/tenure.png) |
| **Monthly Charges** — Churners pay higher monthly charges on average | ![Monthly Charges](images/monthly_charges.png) |
| **Contract Type** — Month-to-month contracts have highest churn rate | ![Contract Type](images/contract_type.png) |
| **Internet Service** — Fiber optic users churn more than DSL | ![Internet Service](images/internet_service.png) |
| **Payment Method** — Electronic check users have highest churn | ![Payment Method](images/payment_method.png) |
| **Senior Citizen** — Senior citizens have higher churn rate | ![Senior Citizen](images/senior_citizen.png) |
| **Gender Distribution** — Churn is similar across genders | ![Gender Distribution](images/gender_distribution.png) |

### Correlation Analysis

![Correlation Heatmap](images/correlation_heatmap.png)

## Model Performance

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

### Feature Importance

The top predictors of churn according to the Random Forest model:

![Feature Importance](images/feature_importance.png)

### SHAP Model Explanations

SHAP values explain individual predictions:

| Explanation Type | Visualization |
|-----------------|--------------|
| **Force Plot** | ![SHAP Force Plot](images/shap_force_plot.png) |
| **Summary Bar** | ![SHAP Summary Bar](images/shap_summary_bar.png) |
| **Beeswarm Summary** | ![SHAP Beeswarm](images/shap_summary_beeswarm.png) |
| **Waterfall Plot** | ![SHAP Waterfall](images/shap_waterfall.png) |

## Results

- **Best Model:** Random Forest (tuned via GridSearchCV)
- **Evaluation Metrics:**
  - Accuracy: ~80%
  - Precision: ~67%
  - Recall: ~55%
  - F1 Score: ~60%
  - ROC-AUC: ~85%

## Live Demo

Run the dashboard locally:
```bash
streamlit run app.py
```

The dashboard includes 6 pages:
1. **Home** — Project overview
2. **Dataset Summary** — Data overview and statistics
3. **KPIs** — Key business metrics
4. **Data Analysis** — Visual exploration of features
5. **Prediction** — Predict churn for a new customer with SHAP explanations
6. **About** — Model details and methodology
