# Business Analysis Report: Customer Churn Prediction

## Executive Summary

> **"Our model catches 49 out of every 100 customers about to churn, with 66% precision."**

Using machine learning on 7,043 customer records, we identified **who leaves, who stays, and why**. The primary churn drivers are **contract type**, **tenure length**, **monthly charges**, and **service adoption**. Below are data-backed retention strategies with measurable impact.

---

## 1. Which Customers Leave?

> **Profile of a "Churn-Prone" Customer**

| Factor | Churner Profile | Statistic |
|--------|----------------|-----------|
| **Contract** | Month-to-month | 88% of churners have month-to-month |
| **Tenure** | Less than 12 months | Avg tenure of churners: **18 months** vs 38 months for loyal |
| **Monthly Charges** | Above $70/month | Avg $74 for churners vs $61 for loyal |
| **Internet Service** | Fiber optic | 41% churn rate vs 19% for DSL |
| **Tech Support** | Does NOT subscribe | 2x higher churn rate without it |
| **Online Security** | Does NOT subscribe | 3x higher churn rate without it |
| **Payment Method** | Electronic check | 45% churn rate — highest of all methods |
| **Senior Citizen** | Yes | 41% churn rate vs 23% for non-seniors |

**Churn Rate by Segment:**

| Segment | Churn Rate |
|---------|-----------|
| Month-to-month contract | **42.7%** |
| One-year contract | 11.3% |
| Two-year contract | 2.9% |
| Fiber optic internet | 41.9% |
| DSL internet | 19.0% |
| Electronic check payment | 45.3% |
| No online security | **49.1%** |
| No tech support | 45.8% |

---

## 2. Which Customers Stay?

> **Profile of a "Loyal Customer"**

| Factor | Loyal Customer Profile |
|--------|----------------------|
| **Contract** | Two-year contract (97% retention) |
| **Tenure** | 5+ years (49+ months) |
| **Monthly Charges** | Below $40/month |
| **Internet Service** | DSL or No internet |
| **Tech Support** | Subscribed |
| **Online Security** | Subscribed |
| **Payment Method** | Credit card or Bank transfer (automatic) |
| **Dependents** | Has dependents |
| **Partner** | Has a partner |

**Retention Rate by Segment:**

| Segment | Retention Rate |
|---------|---------------|
| Two-year contract | **97.1%** |
| Tenure > 48 months | 96.2% |
| Has both Partner + Dependents | 92.0% |
| Has Tech Support | 86.4% |
| Has Online Security | 86.0% |
| Credit card (auto) payment | 93.5% |

---

## 3. Why Do They Leave? (Root Cause Analysis)

### Ranked by Predictive Power (SHAP Analysis)

| Rank | Feature | Impact | Business Meaning |
|------|---------|--------|-----------------|
| **#1** | **Contract** | Highest | Month-to-month = zero commitment = easy to leave |
| **#2** | **Tenure** | Very High | New customers haven't built loyalty or switching costs |
| **#3** | **Monthly Charges** | Very High | High bills drive price-sensitive customers away |
| **#4** | **Service Count** | High | Fewer subscribed services = lower switching costs |
| **#5** | **Internet Service** | High | Fiber optic customers are less satisfied or price-sensitive |
| **#6** | **Online Security** | High | Without it, customers feel unprotected |
| **#7** | **Tech Support** | High | No support = poor experience when issues arise |
| **#8** | **Payment Method** | Medium | Electronic check = manual payment = inconvenient |
| **#9** | **Paperless Billing** | Medium | Automated billing increases retention |
| **#10** | **Senior Citizen** | Medium | Seniors more sensitive to service quality |

### The "Churn Spiral" Pattern

```
Low Tenure (new customer)
    ↓
Month-to-month contract (no commitment)
    ↓
High Monthly Charges (no loyalty discount)
    ↓
No Tech Support / Online Security (poor experience)
    ↓
Fiber optic issues (frustration with service)
    ↓
Electronic check payment (inconvenient billing)
    ↓
                    ★ CHURN
```

---

## 4. Retention Recommendations

### HIGHEST IMPACT (Can reduce churn by 40-50% in targeted segments)

| # | Problem | Target Segment | Solution | Estimated Impact |
|---|---------|---------------|----------|-----------------|
| **1** | Month-to-month churn at 42.7% | 3,850 M2M customers | **Promote annual contracts with a discount** — Offer 1-month free if they switch to annual; auto-enroll after 12 months | Could save **~1,200 customers/year** |
| **2** | Low tenure = low loyalty | Customers < 12 months | **Onboarding loyalty program** — "First 6 months: 10% off monthly charges. Milestone rewards at 3, 6, 12 months." | Could save **~500 new customers/year** |
| **3** | High monthly charges driving churn | Customers > $70/month with risk score > 0.6 | **Targeted loyalty discounts** — "You've been with us X months. Here's 15% off for the next 3 months." | Could save **~400 customers/year** |
| **4** | No tech support = higher churn | 3,100 customers without support | **Free 30-day Tech Support trial** — Auto-convert to paid after trial with opt-out option | Could save **~350 customers/year** |
| **5** | No online security = 3x churn | 3,400 customers without security | **Bundle Online Security free for 3 months** — "Protect your data, on us." | Could save **~300 customers/year** |

### MEDIUM IMPACT (Can reduce churn by 15-25%)

| # | Problem | Solution |
|---|---------|----------|
| **6** | Fiber optic churn at 41.9% | Speed upgrade loyalty offer + price lock for 12 months |
| **7** | Electronic check users churn at 45.3% | Incentivize auto-pay: "Switch to auto-pay, get $5/month off" |
| **8** | Senior citizens churn at 41% | Senior concierge support line + simplified billing |
| **9** | New customers (0-6 months) unstable | "Welcome series" emails + dedicated onboarding call |
| **10** | Customers with 0-1 services | Upsell bundles: Internet + Security + Support for $X/month flat |

### QUICK WINS (Can reduce churn by 5-10%, minimal cost)

| # | Action | Cost | Expected Outcome |
|---|--------|------|-----------------|
| **11** | Send risk alerts to retention team weekly | Minimal | Proactive outreach to high-risk customers |
| **12** | Implement "Save & Win" for retention agents | Low | Incentivize agents to offer targeted discounts |
| **13** | Add churn probability display in CRM | Low | Agents see risk score on every call |
| **14** | Auto-enroll paperless billing with $2 credit | Very Low | Increase auto-pay adoption |
| **15** | SMS/email reminders before contract end | Low | Renewal conversation before they shop around |

---

## 5. Financial Projection

### Current State

| Metric | Value |
|--------|-------|
| Total customers | 7,043 |
| Annual churn rate | ~26.5% |
| Customers lost per year | ~1,866 |
| Avg revenue per customer (monthly) | $64.80 |
| Annual revenue lost to churn | **~$1.45M** |

### With Retention Program (Top 5 Recommendations)

| Metric | After Implementation | Improvement |
|--------|--------------------|-------------|
| Annual churn rate | ~18% | **-8.5%** |
| Customers retained per year | ~600 | +600 customers |
| Revenue saved annually | **~$466,560** | +32% retention ROI |

---

## 6. Implementation Roadmap

| Phase | Timeline | Actions |
|-------|----------|---------|
| **Phase 1: Quick Wins** | Weeks 1-2 | Deploy churn risk scoring in CRM. Set up weekly risk alerts. Start auto-pay incentives. |
| **Phase 2: Outreach** | Weeks 3-6 | Launch free Tech Support trial. Start Online Security bundle promo. Contact high-risk M2M customers. |
| **Phase 3: Retention Programs** | Weeks 7-12 | Roll out loyalty discount program for <12 month customers. Implement senior concierge line. |
| **Phase 4: Monitor & Optimize** | Ongoing | Track churn rate weekly. A/B test retention offers. Retrain model quarterly with new data. |

---

## 7. Model Performance & Trust

Our model predicts churn with **79.9% accuracy** and an **ROC-AUC of 0.84** (strong predictive power).

| Metric | Value | Business Meaning |
|--------|-------|-----------------|
| **Recall** | 49.5% | We catch ~1 of every 2 churners |
| **Precision** | 66.0% | 2 of 3 flagged customers actually churn |
| **F1 Score** | 0.565 | Best balance of precision & recall |

**The model has room to improve.** With better data (usage logs, support tickets, web activity), recall could reach 70-80%.

---

*Report generated from the Customer Churn Prediction ML model — Logistic Regression (F1: 0.565, ROC-AUC: 0.841)*
