"""
data_collector.py
-----------------
ReviewGuard - Phase 1
"""

import sys
import io

# Force UTF-8 encoding for Windows terminal (fixes emoji errors)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

"""
data_collector.py
-----------------
ReviewGuard - Phase 1 (Updated)
Loads 3 Amazon datasets, maps them to a unified schema,
cleans them, and merges into one master file.
"""

import pandas as pd
import numpy as np
import os

# ── File Paths ────────────────────────────────────────────────────────────────

RAW_DIR        = "data/raw"
PROCESSED_FILE = "data/processed/reviews_cleaned.csv"

# ── Schema Definition ─────────────────────────────────────────────────────────
# This is our final unified column structure

SCHEMA = [
    "product_id",
    "product_name",
    "reviewer_id",
    "reviewer_name",
    "rating",
    "review_date",
    "review_text",
    "review_title",
    "verified_purchase",
    "helpful_votes",
    "category",
    "price",
    "source_dataset"          # Tracks which dataset this row came from
]

# ── Process Dataset 1: amazon.csv ────────────────────────────────────────────

def process_amazon1():
    """
    Processes the original Amazon India sales dataset.
    """
    filepath = os.path.join(RAW_DIR, "amazon.csv")
    
    if not os.path.exists(filepath):
        print("  ⚠️ amazon.csv not found, skipping...")
        return None

    print("  Loading amazon.csv...")
    df = pd.read_csv(filepath)

    mapped = pd.DataFrame()
    mapped["product_id"]      = df["product_id"]
    mapped["product_name"]    = df["product_name"]
    mapped["reviewer_id"]     = df["user_id"]
    mapped["reviewer_name"]   = df["user_name"]
    mapped["rating"]          = pd.to_numeric(df["rating"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    mapped["review_date"]     = "2023-01-01"  # Not available in this dataset
    mapped["review_text"]     = df["review_content"].fillna("Unknown")
    mapped["review_title"]    = df["review_title"].fillna("Unknown")
    mapped["verified_purchase"] = "Unknown"
    mapped["helpful_votes"]   = 0
    mapped["category"]        = df["category"]
    mapped["price"]           = df["discounted_price"].astype(str).str.replace("₹", "").str.replace(",", "")
    mapped["price"]           = pd.to_numeric(mapped["price"], errors="coerce")
    mapped["is_recommended"] = False  # Not available in this dataset
    mapped["source_dataset"]  = "amazon1"
    
    mapped["source_dataset"]  = "amazon1"

    print(f"  ✅ Processed {len(mapped)} rows from amazon.csv")
    return mapped

# ── Process Dataset 2: amazon2.csv ───────────────────────────────────────────

def process_amazon2():
    """
    Processes the Datafiniti Amazon product reviews dataset.
    """
    filepath = os.path.join(RAW_DIR, "amazon2.csv")
    
    if not os.path.exists(filepath):
        print("  ⚠️ amazon2.csv not found, skipping...")
        return None

    print("  Loading amazon2.csv (this may take a moment)...")
    # low_memory=False fixes the dtype warning
    df = pd.read_csv(filepath, low_memory=False)

    mapped = pd.DataFrame()
    mapped["product_id"]      = df["id"]
    mapped["product_name"]    = df["name"]
    
    # Generate reviewer_id from reviews.id if available, else from username
    if "reviews.id" in df.columns:
        mapped["reviewer_id"]     = df["reviews.id"].astype(str)
    else:
        mapped["reviewer_id"]     = "RID_" + df["reviews.username"].astype(str).apply(lambda x: str(abs(hash(x)))[:8])
        
    mapped["reviewer_name"]   = df["reviews.username"]
    mapped["rating"]          = pd.to_numeric(df["reviews.rating"], errors="coerce")
    mapped["review_date"]     = pd.to_datetime(df["reviews.date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mapped["review_text"]     = df["reviews.text"].fillna("Unknown")
    mapped["review_title"]    = df["reviews.title"].fillna("Unknown")
    
    # Convert verified purchase to Yes/No/Unknown
    if "reviews.didPurchase" in df.columns:
        mapped["verified_purchase"] = df["reviews.didPurchase"].astype(str).map(
            {"True": "Yes", "False": "No", "nan": "Unknown"}
        ).fillna("Unknown")
    else:
        mapped["verified_purchase"] = "Unknown"


    mapped    # Add is_recommended boolean flag
    if "reviews.doRecommend" in df.columns:
        mapped["is_recommended"] = df["reviews.doRecommend"].map(
            {True: True, "True": True, False: False, "False": False}
        ).fillna(False)
    else:
        mapped["is_recommended"] = False["helpful_votes"]   = df["reviews.numHelpful"].fillna(0).astype(int)
    mapped["category"]        = df["categories"]
    mapped["price"]           = np.nan  # Price not directly available in clean format
    mapped["source_dataset"]  = "amazon2"

    print(f"  ✅ Processed {len(mapped)} rows from amazon2.csv")
    return mapped

# ── Process Dataset 3: amazon3.csv ───────────────────────────────────────────

def process_amazon3():
    """
    Processes the Unlocked Mobile Phones reviews dataset.
    """
    filepath = os.path.join(RAW_DIR, "amazon3.csv")
    
    if not os.path.exists(filepath):
        print("  ⚠️ amazon3.csv not found, skipping...")
        return None

    print("  Loading amazon3.csv (this is a large file, please wait)...")
    df = pd.read_csv(filepath)

    mapped = pd.DataFrame()
    
    # Generate product_id from Product Name hash
    mapped["product_id"]      = "PID_" + df["Product Name"].astype(str).apply(lambda x: str(abs(hash(x)))[:8])
    mapped["product_name"]    = df["Product Name"]
    mapped["reviewer_id"]     = "RID_ANON_" + df.index.astype(str)  # No user info
    mapped["reviewer_name"]   = "Anonymous"
    mapped["rating"]          = pd.to_numeric(df["Rating"], errors="coerce")
    mapped["review_date"]     = np.nan  # Not available
    mapped["review_text"]     = df["Reviews"].fillna("Unknown")
    mapped["review_title"]    = np.nan  # Not available
    mapped["verified_purchase"] = "Unknown"
    mapped["helpful_votes"]   = df["Review Votes"].fillna(0).astype(int)
    mapped["category"]        = df["Brand Name"].fillna("Mobile Phones")
    mapped["price"]           = pd.to_numeric(df["Price"], errors="coerce")
    mapped["is_recommended"] = False  # Not available in this dataset
    mapped["source_dataset"]  = "amazon3"
    
    mapped["source_dataset"]  = "amazon3"

    print(f"  ✅ Processed {len(mapped)} rows from amazon3.csv")
    return mapped

# ── Process Dataset 4: amazon4.csv ───────────────────────────────────────────

def process_amazon4():
    """
    Processes amazon4.csv — Same Datafiniti schema as amazon2.
    Small but 100% clean recommendation signals.
    """
    filepath = os.path.join(RAW_DIR, "amazon4.csv")
    
    if not os.path.exists(filepath):
        print("  ⚠️ amazon4.csv not found, skipping...")
        return None

    print("  Loading amazon4.csv...")
    df = pd.read_csv(filepath, low_memory=False)

    mapped = pd.DataFrame()
    mapped["product_id"]      = df["id"]
    mapped["product_name"]    = df["name"]
    
    # Generate reviewer_id from username (reviews.id is mostly NaN)
    mapped["reviewer_id"] = df["reviews.username"].astype(str).apply(
        lambda x: "RID_" + str(abs(hash(x)))[:10]
    )
    
    mapped["reviewer_name"]   = df["reviews.username"]
    mapped["rating"]          = pd.to_numeric(df["reviews.rating"], errors="coerce")
    mapped["review_date"]     = pd.to_datetime(df["reviews.date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mapped["review_text"]     = df["reviews.text"].fillna("Unknown")
    mapped["review_title"]    = df["reviews.title"].fillna("Unknown")
    
    # doRecommend → verified_purchase mapping
    if "reviews.doRecommend" in df.columns:
        mapped["verified_purchase"] = df["reviews.doRecommend"].astype(str).map(
            {"True": "Yes", "False": "No", "nan": "Unknown"}
        ).fillna("Unknown")
    else:
        mapped["verified_purchase"] = "Unknown"
        # Add is_recommended boolean flag
    if "reviews.doRecommend" in df.columns:
        mapped["is_recommended"] = df["reviews.doRecommend"].map(
            {True: True, "True": True, False: False, "False": False}
        ).fillna(False)
    else:
        mapped["is_recommended"] = False   
    mapped["helpful_votes"]   = df["reviews.numHelpful"].fillna(0).astype(int)
    mapped["category"]        = df["categories"]
    mapped["price"]           = np.nan
    mapped["source_dataset"]  = "amazon4"

    print(f"  ✅ Processed {len(mapped)} rows from amazon4.csv")
    return mapped

# ── Process Dataset 5: amazon5.csv ───────────────────────────────────────────

def process_amazon5():
    """
    Processes amazon5.csv — Same Datafiniti schema as amazon2/amazon4.
    Largest of the new datasets with diverse categories (batteries, etc.)
    """
    filepath = os.path.join(RAW_DIR, "amazon5.csv")
    
    if not os.path.exists(filepath):
        print("  ⚠️ amazon5.csv not found, skipping...")
        return None

    print("  Loading amazon5.csv...")
    df = pd.read_csv(filepath, low_memory=False)

    mapped = pd.DataFrame()
    mapped["product_id"]      = df["id"]
    mapped["product_name"]    = df["name"]
    
    mapped["reviewer_id"] = df["reviews.username"].astype(str).apply(
        lambda x: "RID_" + str(abs(hash(x)))[:10]
    )
    
    mapped["reviewer_name"]   = df["reviews.username"]
    mapped["rating"]          = pd.to_numeric(df["reviews.rating"], errors="coerce")
    mapped["review_date"]     = pd.to_datetime(df["reviews.date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mapped["review_text"]     = df["reviews.text"].fillna("Unknown")
    mapped["review_title"]    = df["reviews.title"].fillna("Unknown")
    
    if "reviews.doRecommend" in df.columns:
        mapped["verified_purchase"] = df["reviews.doRecommend"].astype(str).map(
            {"True": "Yes", "False": "No", "nan": "Unknown"}
        ).fillna("Unknown")
    else:
        mapped["verified_purchase"] = "Unknown"
        # Add is_recommended boolean flag
    if "reviews.doRecommend" in df.columns:
        mapped["is_recommended"] = df["reviews.doRecommend"].map(
            {True: True, "True": True, False: False, "False": False}
        ).fillna(False)
    else:
        mapped["is_recommended"] = False

    mapped["helpful_votes"]   = df["reviews.numHelpful"].fillna(0).astype(int)
    mapped["category"]        = df["categories"]
    mapped["price"]           = np.nan
    mapped["source_dataset"]  = "amazon5"

    print(f"  ✅ Processed {len(mapped)} rows from amazon5.csv")
    return mapped

# ── Clean Merged Data ─────────────────────────────────────────────────────────

def clean_merged_data(df):
    """
    Performs final cleaning on the merged dataset.
    """
    print("\n" + "=" * 55)
    print("  CLEANING MERGED DATA")
    print("=" * 55)

    original_count = len(df)

    # 1. Remove rows where rating is completely missing (essential column)
    df = df.dropna(subset=["rating"])
    print(f"  ✅ Dropped {original_count - len(df)} rows with missing ratings")

    # 2. Ensure rating is between 1 and 5
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]
    print(f"  ✅ Filtered ratings to 1-5 range")

    # 3. Remove exact duplicate review texts per product
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["product_id", "review_text"])
    print(f"  ✅ Removed {before_dedup - len(df)} duplicate reviews per product")

    # 4. Standardize verified_purchase
    df["verified_purchase"] = df["verified_purchase"].fillna("Unknown")

    # 5. Reset index
    df = df.reset_index(drop=True)

    return df

# ── Fix Data Issues ───────────────────────────────────────────────────────────

def fix_data_issues(df):
    """
    Fixes all data quality issues found in the quality check:
    1. Rounds decimal ratings to nearest whole number
    2. Fills missing reviewer IDs
    3. Handles missing dates properly
    4. Flags duplicate review texts
    5. Fixes missing product names
    """
    print("\n" + "=" * 55)
    print("  FIXING DATA ISSUES")
    print("=" * 55)

    # ── Fix 1: Round decimal ratings ─────────────────────────────────────────
    before = df["rating"].nunique()
    df["rating"] = df["rating"].round().astype(int)
    after = df["rating"].nunique()
    print(f"  ✅ Rounded decimal ratings")
    print(f"     Unique rating values: {before} → {after}")

    # ── Fix 2: Fill missing reviewer IDs ─────────────────────────────────────
    missing_ids = df["reviewer_id"].isna().sum()
    df["reviewer_id"] = df.apply(
        lambda row: "RID_" + str(abs(hash(str(row["reviewer_name"]) + 
                                         str(row["product_id"]))))[:10]
        if pd.isna(row["reviewer_id"]) else row["reviewer_id"],
        axis=1
    )
    print(f"  ✅ Filled {missing_ids} missing reviewer IDs")

    # ── Fix 3: Handle missing dates ───────────────────────────────────────────
    # Instead of leaving NaN or wrong dates
    # We mark them clearly as "Unknown"
    df["review_date"] = df["review_date"].fillna("Unknown")
    df["review_date"] = df["review_date"].astype(str)
    
    # Fix the 2 reviews before 2013
    df.loc[df["review_date"] < "2013-01-01", "review_date"] = "Unknown"
    print(f"  ✅ Handled missing and invalid dates")

    # ── Fix 4: Flag duplicate review texts ───────────────────────────────────
    # Instead of removing them (they might be real)
    # We add a flag column for the ML model to use later
    df["is_duplicate_text"] = df.duplicated(
        subset=["review_text"], keep=False
    ).astype(int)
    duplicate_count = df["is_duplicate_text"].sum()
    print(f"  ✅ Flagged {duplicate_count} duplicate review texts")
    print(f"     (Flag added as 'is_duplicate_text' column)")

    # ── Fix 5: Fill missing product names ────────────────────────────────────
    missing_names = df["product_name"].isna().sum()
    df["product_name"] = df["product_name"].fillna("Unknown Product")
    print(f"  ✅ Filled {missing_names} missing product names")

    # ── Fix 6: Fill missing reviewer names ───────────────────────────────────
    df["reviewer_name"] = df["reviewer_name"].fillna("Anonymous")
    print(f"  ✅ Filled missing reviewer names")

    # ── Fix 7: Fill missing review titles ────────────────────────────────────
    df["review_title"] = df["review_title"].fillna("No Title")
    print(f"  ✅ Filled missing review titles")

    # ── Fix 8: Add review length column ──────────────────────────────────────
    df["review_length"] = df["review_text"].astype(str).str.len()
    print(f"  ✅ Added review_length column")

    # ── Fix 9: Add short review flag ─────────────────────────────────────────
    # Reviews under 10 characters are suspicious
    df["is_short_review"] = (df["review_length"] < 10).astype(int)
    short_count = df["is_short_review"].sum()
    print(f"  ✅ Flagged {short_count} suspiciously short reviews")

    print(f"\n  Final shape: {df.shape}")
    print(f"  Final columns: {list(df.columns)}")

    return df

# ── Main Function ─────────────────────────────────────────────────────────────

def main():
    print("\n🚀 ReviewGuard - Multi-Source Data Collector Starting...\n")

    # Process all 3 datasets
    print("=" * 55)
    print("  PROCESSING DATASETS")
    print("=" * 55)

    df1 = process_amazon1()
    df2 = process_amazon2()
    df3 = process_amazon3()
    df4 = process_amazon4()
    df5 = process_amazon5()

    # Combine all datasets
    print("\n" + "=" * 55)
    print("  MERGING DATASETS")
    print("=" * 55)

    frames = [df for df in [df1, df2, df3, df4, df5] if df is not None]
    
    if not frames:
        print("❌ No datasets found! Please add CSV files to data/raw/")
        return

    df_merged = pd.concat(frames, ignore_index=True)
    print(f"  ✅ Combined row count: {len(df_merged)}")

        # Clean the merged data
    df_clean = clean_merged_data(df_merged)

    # Fix all data quality issues
    df_clean = fix_data_issues(df_clean)

    # Save
    print("\n" + "=" * 55)
    print("  SAVING PROCESSED DATA")
    print("=" * 55)

    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df_clean.to_csv(PROCESSED_FILE, index=False, encoding="utf-8")
    print(f"  ✅ Saved to: {PROCESSED_FILE}")

    # Final Stats
    print("\n" + "=" * 55)
    print("  FINAL DATASET STATS")
    print("=" * 55)
    print(f"  Total reviews     : {len(df_clean):,}")
    print(f"  Unique products   : {df_clean['product_id'].nunique():,}")
    print(f"  Unique reviewers  : {df_clean['reviewer_id'].nunique():,}")
    print(f"  Average rating    : {df_clean['rating'].mean():.2f} ⭐")
    print(f"\n  Source breakdown  :")
    print(df_clean["source_dataset"].value_counts().to_string())

    print("\n✅ data_collector.py completed successfully!")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()