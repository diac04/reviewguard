# 🛡️ ReviewGuard

### Multi-Signal Fake Review Detection System for Amazon

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reviewguard-diac04.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6F16)](https://xgboost.ai)

Combines **statistical analysis**, **machine learning**, and **graph theory** to detect coordinated fake reviews on Amazon products. Analyzes 59K reviews across 89 products, identifies fake reviewer networks, and produces per-product trust scores.

**🚀 [Try the Live Demo](https://reviewguard-diac04.streamlit.app)**


---

## 📸 Dashboard Preview

### Landing Page
![Landing Page](dashboard/screenshots/landingpage.png)

### Overview — Executive Summary & Analytics
![Overview](dashboard/screenshots/overview1.png)
![Overview Continued](dashboard/screenshots/overview2.png)

### Product Explorer — Filter & Search 89 Products
![Product Explorer](dashboard/screenshots/productexplorer.png)

### Product Details — Deep-Dive Analysis
![Product Details](dashboard/screenshots/prpductdetails1.png)
![Product Details Continued](dashboard/screenshots/productdetails2.png)

### Methodology — Multi-Signal Approach
![Methodology](dashboard/screenshots/methodology1.png)
![Methodology Continued](dashboard/screenshots/methodology2.png)
---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| Reviews Analyzed | **59,604** |
| Products Scored | **89** |
| Fake Review Rings Detected | **3** |
| Suspicious Reviewers Identified | **82** |
| Ground Truth Reliability (Cohen's κ) | **0.92** |
| Critical Risk Products Flagged | **5** |

---

## 🎯 The Approach

Three complementary detection techniques combined into a unified trust score:

### 📊 Phase 2 — Statistical Analysis (30% weight)
Chi-square tests, Mann-Whitney U tests, review burst detection, temporal pattern analysis

### 🤖 Phase 3 — Machine Learning (30% weight)
XGBoost classifier with 126 features (TF-IDF + behavioral), SMOTE class balancing, 5-fold stratified cross-validation

### 🕸️ Phase 4 — Network Analysis (40% weight)
Bipartite reviewer-product graph, Louvain community detection, coordinated behavior identification

Suspicion Score = 0.30 × Phase2 + 0.30 × Phase3 + 0.40 × Phase4
Trust Score = 100 - Suspicion Score


---

## 🔥 Key Findings

- **Convergent validation:** Same suspicious products flagged by 2 independent methods (temporal + network)
- **"Generic name pattern":** Fake reviewer rings dominated by names like Mike, Dave, Nick, Chris — a documented signature of paid review farms
- **Statistical proof:** Rating distribution is non-uniform with p ≈ 0 (chi-square test)
- **Rating inflation detected:** Up to 1.12⭐ artificial boost on flagged products

---

## 🛠️ Tech Stack

**Core:** Python 3.11, pandas, numpy, scipy  
**ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)  
**Graph Analysis:** NetworkX, python-louvain  
**Visualization:** Plotly, matplotlib, seaborn  
**Dashboard:** Streamlit, Streamlit Cloud

---

## 💼 Business Value

Designed as an **internal fraud detection tool** for platforms like Amazon:
- Identifies fake reviewer networks for account bans
- Flags high-risk products for moderator review
- Supports regulatory compliance (FTC, EU DSA)
- Estimated ROI: **$25 saved per $1 invested** (returns, fines, churn)

---

## 🎓 Notable Technical Insights

- **Discovered SMOTE overfitting** via CV-vs-test gap analysis (0.71 → 0.40 F1)
- **Identified label leakage risk** and mitigated with orthogonal network features
- **Systematic threshold selection** in graph analysis (tested 4-8 shared products)
- **Iterative dataset expansion** improved ML performance by 122% (F1: 0.18 → 0.40)

---

## 📁 Project Structure
```
reviewguard/
├── dashboard/
│ └── app.py # Streamlit dashboard
├── data/
│ ├── processed/ # Cleaned datasets
│ └── graphs/ # Network analysis outputs
├── notebooks/
│ ├── Phase2_EDA.ipynb # Statistical analysis
│ ├── Phase3_ML_Modeling.ipynb # ML classification
│ ├── Phase4_Network.ipynb # Graph analysis
│ └── Phase5_Trust_Score.ipynb # Integration
├── src/
│ ├── data_collector.py # ETL pipeline
│ └── data_quality.py # Quality checks
├── outputs/
│ └── dashboard/ # Dashboard data files
├── requirements.txt
└── README.md
```


---

## 🚀 Quick Start

```
# Clone and install
git clone https://github.com/diac04/reviewguard.git
cd reviewguard
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/app.py
```

## 📊 Data Sources
Public Amazon review datasets from Kaggle:

Datafiniti Amazon Consumer Reviews
Amazon India Sales Dataset
AmazonBasics Product Reviews

## 👤 Author
Diyashi Roy
- Data Analyst Portfolio Project
- LinkedIn: https://www.linkedin.com/in/diyashi-roy-894811290/
- GitHub: https://github.com/diac04

<div align="center">
🛡️ Built with rigor, honesty, and a commitment to real-world data science
If you find this project useful, please ⭐ star the repository!

</div> 