"""
╔══════════════════════════════════════════════════════════════╗
║  ReviewGuard — Interactive Trust Score Dashboard            ║
║  Built with Streamlit + Plotly                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReviewGuard — Amazon Fake Review Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3498db 0%, #e74c3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subheader {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    .stMetric label {
        color: #2c3e50;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all dashboard data (cached for performance)."""
    products = pd.read_csv("outputs/dashboard/product_trust_report.csv")
    reviews = pd.read_csv("outputs/dashboard/review_details.csv", low_memory=False)
    with open("outputs/dashboard/summary_stats.json", "r") as f:
        stats = json.load(f)
    return products, reviews, stats

try:
    products_df, reviews_df, stats = load_data()
except FileNotFoundError:
    st.error("⚠️ Dashboard data not found. Please run Phase 5 notebook first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🛡️ ReviewGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Amazon Fake Review Detection & Trust Analytics</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🔍 Filters")
    
    # Trust level filter
    trust_levels = st.multiselect(
        "Trust Level",
        options=["HIGH TRUST", "MEDIUM TRUST", "LOW TRUST", "CRITICAL RISK"],
        default=["HIGH TRUST", "MEDIUM TRUST", "LOW TRUST", "CRITICAL RISK"]
    )
    
    # Trust score range
    trust_range = st.slider(
        "Trust Score Range",
        min_value=0.0, max_value=100.0,
        value=(0.0, 100.0),
        step=5.0
    )
    
    # Min reviews filter
    min_reviews = st.number_input(
        "Minimum Reviews",
        min_value=1, max_value=1000,
        value=5
    )
    
    # Search
    search_term = st.text_input("🔎 Search Products", "")
    
    st.markdown("---")
    st.markdown("### 📊 About ReviewGuard")
    st.info(
        "Multi-signal fake review detection system combining:\n\n"
        "• **Statistical analysis** (Phase 2)\n"
        "• **ML classifier** (Phase 3)\n"
        "• **Network analysis** (Phase 4)\n\n"
        "Analyzes 59K reviews across 89 Amazon products."
    )

# Apply filters
filtered_df = products_df[
    (products_df["trust_level"].isin(trust_levels)) &
    (products_df["trust_score"] >= trust_range[0]) &
    (products_df["trust_score"] <= trust_range[1]) &
    (products_df["total_reviews"] >= min_reviews)
]

if search_term:
    filtered_df = filtered_df[
        filtered_df["product_name"].str.contains(search_term, case=False, na=False)
    ]

# ══════════════════════════════════════════════════════════════════════════════
#  KEY METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Products",
        f"{stats['total_products']}",
        f"Analyzed"
    )

with col2:
    st.metric(
        "Total Reviews",
        f"{stats['total_reviews']:,}",
        f"Scored"
    )

with col3:
    st.metric(
        "Suspicious Reviews",
        f"{stats['suspicious_reviews']:,}",
        f"{stats['suspicion_rate']}% flagged",
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Critical Risk Products",
        f"{stats['critical_products']}",
        f"Need investigation",
        delta_color="inverse"
    )

with col5:
    st.metric(
        "Avg Trust Score",
        f"{stats['avg_trust_score']}/100",
        "Platform average"
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "🔍 Product Explorer", 
    "📋 Product Details",
    "🎯 Methodology"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Trust Level Distribution
        trust_counts = products_df["trust_level"].value_counts().reindex(
            ["HIGH TRUST", "MEDIUM TRUST", "LOW TRUST", "CRITICAL RISK"]
        ).fillna(0)
        
        color_map = {
            "HIGH TRUST": "#2ecc71",
            "MEDIUM TRUST": "#f1c40f",
            "LOW TRUST": "#e67e22",
            "CRITICAL RISK": "#e74c3c"
        }
        colors = [color_map[level] for level in trust_counts.index]
        
        fig_trust = go.Figure(data=[go.Pie(
            labels=trust_counts.index,
            values=trust_counts.values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textfont=dict(size=14, color="white"),
            textinfo="label+percent"
        )])
        fig_trust.update_layout(
            title=dict(text="Trust Level Distribution", font=dict(size=18)),
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_trust, use_container_width=True)
    
    with col2:
        # Trust Score Distribution
        fig_hist = px.histogram(
            products_df,
            x="trust_score",
            nbins=20,
            color_discrete_sequence=["#3498db"],
            title="Trust Score Distribution"
        )
        fig_hist.add_vline(x=40, line_dash="dash", line_color="red", annotation_text="Critical threshold")
        fig_hist.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Low threshold")
        fig_hist.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="High threshold")
        fig_hist.update_layout(
            xaxis_title="Trust Score (0-100)",
            yaxis_title="Number of Products",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Rating Inflation Analysis
    st.markdown("### 📉 Rating Inflation Analysis")
    st.markdown("*How much have fake reviews inflated product ratings?*")
    
    inflation_df = products_df[products_df["rating_inflation"] > 0].sort_values("rating_inflation", ascending=False).head(15)
    
    if len(inflation_df) > 0:
        fig_inflation = go.Figure()
        fig_inflation.add_trace(go.Bar(
            y=inflation_df["product_name"].str[:50],
            x=inflation_df["avg_rating"],
            name="Displayed Rating",
            orientation="h",
            marker=dict(color="#e74c3c"),
            text=inflation_df["avg_rating"].round(2),
            textposition="inside"
        ))
        fig_inflation.add_trace(go.Bar(
            y=inflation_df["product_name"].str[:50],
            x=inflation_df["adjusted_rating"],
            name="Adjusted Rating",
            orientation="h",
            marker=dict(color="#2ecc71"),
            text=inflation_df["adjusted_rating"].round(2),
            textposition="inside"
        ))
        fig_inflation.update_layout(
            barmode="group",
            title="Top 15 Products: Displayed vs Adjusted Ratings",
            xaxis_title="Rating",
            yaxis_title="Product",
            height=600,
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
        )
        st.plotly_chart(fig_inflation, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: PRODUCT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(f"### 🔍 Filtered Products ({len(filtered_df)} of {len(products_df)})")
    
    # Sortable table
    display_df = filtered_df[[
        "product_name", "trust_score", "trust_level",
        "avg_rating", "adjusted_rating", "rating_inflation",
        "total_reviews", "suspicious_count", "pct_suspicious"
    ]].copy()
    
    display_df.columns = [
        "Product Name", "Trust Score", "Trust Level",
        "Rating", "Adjusted", "Inflation",
        "Reviews", "Suspicious", "% Suspicious"
    ]
    
    # Color coding function
    def color_trust(val):
        if val == "HIGH TRUST":
            return "background-color: #d5f4e6; color: #27ae60"
        elif val == "MEDIUM TRUST":
            return "background-color: #fef5e7; color: #d68910"
        elif val == "LOW TRUST":
            return "background-color: #fdebd0; color: #e67e22"
        elif val == "CRITICAL RISK":
            return "background-color: #fadbd8; color: #c0392b"
        return ""
    
    styled_df = display_df.style.map(color_trust, subset=["Trust Level"])
    st.dataframe(styled_df, use_container_width=True, height=500)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="reviewguard_filtered.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: INDIVIDUAL PRODUCT DETAILS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🔬 Deep Dive Into Individual Product")
    
    # Product selector
    product_names = ["Select a product..."] + filtered_df["product_name"].tolist()
    selected_name = st.selectbox("Choose a product to analyze:", product_names)
    
    if selected_name != "Select a product...":
        product_row = filtered_df[filtered_df["product_name"] == selected_name].iloc[0]
        
        # Product header
        st.markdown(f"## {selected_name}")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trust Score", f"{product_row['trust_score']:.1f}/100", product_row["trust_level"])
        col2.metric("Displayed Rating", f"{product_row['avg_rating']:.2f}⭐")
        col3.metric("Adjusted Rating", f"{product_row['adjusted_rating']:.2f}⭐", 
                    f"-{product_row['rating_inflation']:.2f}⭐" if product_row['rating_inflation'] > 0 else "No change")
        col4.metric("Total Reviews", f"{int(product_row['total_reviews']):,}")
        
        st.markdown("---")
        
        # Signal breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Multi-Signal Breakdown")
            signals = pd.DataFrame({
                "Signal": ["Phase 2 (Statistical)", "Phase 3 (ML)", "Phase 4 (Network)"],
                "Suspicion Score": [
                    product_row["phase2_score"],
                    product_row["phase3_score"],
                    product_row["phase4_score"]
                ]
            })
            
            fig_signals = go.Figure(go.Bar(
                x=signals["Signal"],
                y=signals["Suspicion Score"],
                marker=dict(color=["#3498db", "#9b59b6", "#e74c3c"]),
                text=signals["Suspicion Score"].round(1),
                textposition="outside"
            ))
            fig_signals.update_layout(
                yaxis_title="Suspicion Score",
                yaxis_range=[0, 110],
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_signals, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Review Status")
            review_status = pd.DataFrame({
                "Status": ["Clean Reviews", "Suspicious Reviews"],
                "Count": [
                    product_row["clean_review_count"],
                    product_row["suspicious_count"]
                ]
            })
            
            fig_status = go.Figure(data=[go.Pie(
                labels=review_status["Status"],
                values=review_status["Count"],
                hole=0.4,
                marker=dict(colors=["#2ecc71", "#e74c3c"])
            )])
            fig_status.update_layout(height=350)
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Sample suspicious reviews
        st.markdown("#### 🚨 Sample Suspicious Reviews from This Product")
        product_reviews = reviews_df[reviews_df["product_name"] == selected_name]
        suspicious_reviews = product_reviews[product_reviews["is_suspicious_review"] == 1].head(5)
        
        if len(suspicious_reviews) > 0:
            for idx, review in suspicious_reviews.iterrows():
                with st.expander(f"⚠️ {review['reviewer_name']} — {review['rating']}⭐ (ML Fake Prob: {review['ml_fake_probability']:.2f})"):
                    st.write(f"**Review:** {review['review_text']}")
                    st.write(f"**Date:** {review['review_date']}")
                    st.write(f"**Recommended:** {'Yes' if review['is_recommended'] else 'No'}")
        else:
            st.info("No suspicious reviews found for this product.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 🔬 ReviewGuard Methodology")
    
    st.markdown("""
    ReviewGuard uses a **multi-signal ensemble approach** combining three independent 
    detection techniques. Each phase catches different fraud patterns.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Phase 2: Statistical Analysis**
        
        Weight: **30%**
        
        - Chi-square rating tests
        - Mann-Whitney U tests
        - Burst detection (5+ reviews in 48h)
        - Rating deviation analysis
        - Product-level heuristics
        
        Detects: Statistical anomalies in ratings and timing
        """)
    
    with col2:
        st.info("""
        **🤖 Phase 3: Machine Learning**
        
        Weight: **30%**
        
        - XGBoost classifier
        - 126 features (behavioral + TF-IDF)
        - SMOTE class balancing
        - 5-fold cross-validation
        - F1 = 0.40 on test set
        
        Detects: Text patterns of deceptive reviews
        """)
    
    with col3:
        st.info("""
        **🕸️ Phase 4: Network Analysis**
        
        Weight: **40%**
        
        - Bipartite reviewer-product graph
        - Louvain community detection
        - Statistical threshold (6+ shared)
        - 82 suspicious reviewers found
        - 3 fake review rings detected
        
        Detects: Coordinated reviewer behavior
        """)
    
    st.markdown("---")
    
    st.markdown("### 📐 Trust Score Formula")
    st.latex(r"""
    \text{Suspicion Score} = 0.30 \times \text{Phase2} + 0.30 \times \text{Phase3} + 0.40 \times \text{Phase4}
    """)
    st.latex(r"""
    \text{Trust Score} = 100 - \text{Suspicion Score}
    """)
    
    st.markdown("### 🎯 Trust Level Classification")
    st.markdown("""
    | Trust Score | Trust Level | Meaning |
    |-------------|-------------|---------|
    | 80-100 | 🟢 **HIGH TRUST** | Reviews appear authentic |
    | 60-79 | 🟡 **MEDIUM TRUST** | Normal patterns, no red flags |
    | 40-59 | 🟠 **LOW TRUST** | Some manipulation likely |
    | 0-39 | 🔴 **CRITICAL RISK** | Investigate immediately |
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        Built with Streamlit • Powered by XGBoost + NetworkX + Plotly<br>
        <b>ReviewGuard v1.0</b> — Portfolio Project by Diya
    </div>
    """,
    unsafe_allow_html=True
)