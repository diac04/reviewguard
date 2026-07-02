"""
data_quality.py
---------------
ReviewGuard - Phase 1
This script runs quality checks on our cleaned dataset and flags:
- Missing critical fields
- Duplicate review entries
- Suspicious reviewer IDs (bot-like patterns)
- Date range validation
- Rating anomalies
"""

import pandas as pd
import numpy as np
import os

# ── File Paths ────────────────────────────────────────────────────────────────

PROCESSED_FILE = "data/processed/reviews_cleaned.csv"
REPORT_FILE    = "outputs/reports/data_quality_report.txt"

# ── Helper: Print Section Header ─────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

# ── Check 1: Missing Critical Fields ─────────────────────────────────────────

def check_missing_fields(df):
    """
    Checks for missing values in critical columns.
    Flags any column that has missing values.
    """
    print_header("CHECK 1: Missing Critical Fields")

    # These are the columns that MUST have values
    critical_columns = [
        "product_id",
        "product_name",
        "reviewer_id",
        "reviewer_name",
        "rating",
        "review_text"
    ]

    issues_found = 0

    for col in critical_columns:
        if col not in df.columns:
            print(f"  ❌ Column '{col}' is MISSING from dataset entirely!")
            issues_found += 1
            continue

        missing_count = df[col].isna().sum()
        missing_pct   = (missing_count / len(df)) * 100

        if missing_count > 0:
            print(f"  ⚠️  '{col}' has {missing_count} missing values ({missing_pct:.1f}%)")
            issues_found += 1
        else:
            print(f"  ✅  '{col}' is complete — no missing values")

    print(f"\n  Total issues found: {issues_found}")
    return issues_found

# ── Check 2: Duplicate Reviews ────────────────────────────────────────────────

def check_duplicates(df):
    """
    Checks for duplicate reviews.
    A duplicate is when the same reviewer reviewed
    the same product more than once.
    """
    print_header("CHECK 2: Duplicate Review Entries")

    issues_found = 0

    # Check 2A: Fully duplicate rows (every column is the same)
    full_duplicates = df.duplicated().sum()
    if full_duplicates > 0:
        print(f"  ⚠️  Found {full_duplicates} fully duplicate rows")
        issues_found += full_duplicates
    else:
        print(f"  ✅  No fully duplicate rows found")

    # Check 2B: Same reviewer reviewing same product twice
    if "reviewer_id" in df.columns and "product_id" in df.columns:
        subset_duplicates = df.duplicated(
            subset=["reviewer_id", "product_id"]
        ).sum()

        if subset_duplicates > 0:
            print(f"  ⚠️  Found {subset_duplicates} cases where same reviewer")
            print(f"      reviewed same product more than once")
            issues_found += subset_duplicates
        else:
            print(f"  ✅  No reviewer reviewed same product twice")

    # Check 2C: Identical review text (copy-paste reviews)
    if "review_text" in df.columns:
        # Ignore "Unknown" and short texts
        real_reviews = df[
            (df["review_text"] != "Unknown") &
            (df["review_text"].str.len() > 10)
        ]
        duplicate_texts = real_reviews.duplicated(subset=["review_text"]).sum()

        if duplicate_texts > 0:
            print(f"  ⚠️  Found {duplicate_texts} reviews with IDENTICAL text")
            print(f"      (possible copy-paste fake reviews!)")
            issues_found += duplicate_texts
        else:
            print(f"  ✅  No copy-paste duplicate review texts found")

    print(f"\n  Total issues found: {issues_found}")
    return issues_found

# ── Check 3: Suspicious Reviewer IDs ─────────────────────────────────────────

def check_suspicious_reviewers(df):
    """
    Flags reviewers that show bot-like behavior:
    - Reviewer ID looks auto-generated (all numbers, very short etc)
    - Reviewer name looks fake (single character, numbers in name)
    - Same reviewer reviewed too many products
    """
    print_header("CHECK 3: Suspicious Reviewer IDs (Bot Detection)")

    if "reviewer_id" not in df.columns:
        print("  ❌ reviewer_id column not found")
        return 0

    issues_found    = 0
    suspicious_ids  = []

    # Check 3A: Reviewer IDs that are purely numeric
    numeric_ids = df[
        df["reviewer_id"].astype(str).str.match(r"^\d+$")
    ]
    if len(numeric_ids) > 0:
        print(f"  ⚠️  Found {len(numeric_ids)} purely numeric reviewer IDs")
        print(f"      (these may be auto-generated bot accounts)")
        suspicious_ids.extend(numeric_ids["reviewer_id"].tolist())
        issues_found += len(numeric_ids)
    else:
        print(f"  ✅  No purely numeric reviewer IDs found")

    # Check 3B: Reviewer IDs that are very short (less than 3 chars)
    short_ids = df[
        df["reviewer_id"].astype(str).str.len() < 3
    ]
    if len(short_ids) > 0:
        print(f"  ⚠️  Found {len(short_ids)} very short reviewer IDs")
        issues_found += len(short_ids)
    else:
        print(f"  ✅  No suspiciously short reviewer IDs found")

    # Check 3C: Reviewers who reviewed TOO many products
    # A real person rarely reviews more than 5-10 products
    if "reviewer_id" in df.columns and "product_id" in df.columns:
        reviews_per_reviewer = df.groupby("reviewer_id")["product_id"].count()
        high_volume          = reviews_per_reviewer[reviews_per_reviewer > 5]

        if len(high_volume) > 0:
            print(f"\n  ⚠️  Found {len(high_volume)} reviewers who reviewed")
            print(f"      more than 5 products (possible fake review farms)")
            print(f"\n      Top 5 most active reviewers:")
            top5 = high_volume.sort_values(ascending=False).head(5)
            for reviewer_id, count in top5.items():
                print(f"        Reviewer: {reviewer_id} → {count} reviews")
            issues_found += len(high_volume)
        else:
            print(f"  ✅  No high-volume suspicious reviewers found")

    # Check 3D: Reviewer names with numbers in them
    if "reviewer_name" in df.columns:
        numeric_names = df[
            df["reviewer_name"].astype(str).str.contains(
                r"\d", regex=True, na=False
            )
        ]
        if len(numeric_names) > 0:
            print(f"\n  ⚠️  Found {len(numeric_names)} reviewer names")
            print(f"      containing numbers (e.g. 'User1234')")
            print(f"      Sample names:")
            samples = numeric_names["reviewer_name"].head(5).tolist()
            for name in samples:
                print(f"        - {name}")
            issues_found += len(numeric_names)
        else:
            print(f"  ✅  No numeric reviewer names found")

    print(f"\n  Total issues found: {issues_found}")
    return issues_found

# ── Check 4: Rating Anomalies ─────────────────────────────────────────────────

def check_rating_anomalies(df):
    """
    Checks for suspicious rating patterns:
    - Ratings outside 1-5 range
    - Products with ALL 5-star ratings (suspiciously perfect)
    - Products with ALL 1-star ratings (possible attack)
    - Extreme rating distributions
    """
    print_header("CHECK 4: Rating Anomalies")

    if "rating" not in df.columns:
        print("  ❌ rating column not found")
        return 0

    issues_found = 0

    # Check 4A: Ratings outside valid range (should be 1 to 5)
    invalid_ratings = df[
        (df["rating"] < 1) | (df["rating"] > 5)
    ]
    if len(invalid_ratings) > 0:
        print(f"  ⚠️  Found {len(invalid_ratings)} ratings outside 1-5 range")
        issues_found += len(invalid_ratings)
    else:
        print(f"  ✅  All ratings are within valid range (1-5)")

    # Check 4B: Rating distribution
    rating_dist = df["rating"].value_counts().sort_index()
    print(f"\n  📊 Rating Distribution:")
    for rating, count in rating_dist.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"     {rating}★  {bar} {count} ({pct:.1f}%)")

    # Check 4C: Products with ALL 5-star ratings
    if "product_id" in df.columns:
        product_ratings = df.groupby("product_id")["rating"]

        all_five_star = product_ratings.filter(
            lambda x: (x == 5).all() and len(x) >= 3
        )
        if len(all_five_star) > 0:
            suspicious_products = df[
                df.index.isin(all_five_star.index)
            ]["product_id"].nunique()
            print(f"\n  ⚠️  Found {suspicious_products} products with")
            print(f"      ALL 5-star ratings (3+ reviews)")
            print(f"      (possible fake review boosting!)")
            issues_found += suspicious_products
        else:
            print(f"\n  ✅  No products with suspiciously all 5-star ratings")

    # Check 4D: Overall average rating
    avg_rating = df["rating"].mean()
    print(f"\n  📊 Overall average rating: {avg_rating:.2f}")
    if avg_rating > 4.5:
        print(f"  ⚠️  Average rating is very high (> 4.5)")
        print(f"      This could indicate rating manipulation")
        issues_found += 1
    else:
        print(f"  ✅  Average rating looks normal")

    print(f"\n  Total issues found: {issues_found}")
    return issues_found

# ── Check 5: Date Validation ──────────────────────────────────────────────────

def check_date_validation(df):
    """
    Validates the review_date column:
    - Checks for invalid date formats
    - Checks for future dates
    - Checks for very old dates
    """
    print_header("CHECK 5: Date Range Validation")

    if "review_date" not in df.columns:
        print("  ❌ review_date column not found")
        return 0

    issues_found = 0

    # Try to convert to datetime
    df["review_date_parsed"] = pd.to_datetime(
        df["review_date"], errors="coerce"
    )

    # Check for invalid dates that could not be parsed
    invalid_dates = df["review_date_parsed"].isna().sum()
    if invalid_dates > 0:
        print(f"  ⚠️  Found {invalid_dates} invalid/unparseable dates")
        issues_found += invalid_dates
    else:
        print(f"  ✅  All dates are valid format")

    # Check for future dates (reviews from the future = suspicious)
    future_dates = df[
        df["review_date_parsed"] > pd.Timestamp.now()
    ]
    if len(future_dates) > 0:
        print(f"  ⚠️  Found {len(future_dates)} reviews with FUTURE dates!")
        print(f"      (this is a strong indicator of fake reviews)")
        issues_found += len(future_dates)
    else:
        print(f"  ✅  No future-dated reviews found")

    # Check for very old dates (before Amazon India launched = 2013)
    very_old_dates = df[
        df["review_date_parsed"] < pd.Timestamp("2013-01-01")
    ]
    if len(very_old_dates) > 0:
        print(f"  ⚠️  Found {len(very_old_dates)} reviews before 2013")
        print(f"      (Amazon India launched in 2013, these seem invalid)")
        issues_found += len(very_old_dates)
    else:
        print(f"  ✅  No reviews before Amazon India launch (2013)")

    # Drop the helper column
    df.drop(columns=["review_date_parsed"], inplace=True)

    print(f"\n  Total issues found: {issues_found}")
    return issues_found

# ── Generate Report ───────────────────────────────────────────────────────────

def generate_report(results):
    """
    Saves a summary report of all quality checks.
    """
    print_header("GENERATING QUALITY REPORT")

    os.makedirs("outputs/reports", exist_ok=True)

    total_issues = sum(results.values())

    report = []
    report.append("=" * 55)
    report.append("  ReviewGuard — Data Quality Report")
    report.append("=" * 55)
    report.append("")

    for check_name, issue_count in results.items():
        status = "✅ PASS" if issue_count == 0 else "⚠️  ISSUES FOUND"
        report.append(f"  {check_name}")
        report.append(f"  Status : {status}")
        report.append(f"  Issues : {issue_count}")
        report.append("")

    report.append("-" * 55)
    report.append(f"  TOTAL ISSUES FOUND: {total_issues}")
    report.append("=" * 55)

    # Print to terminal
    for line in report:
        print(line)

    # Save to file
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n  ✅ Report saved to: {REPORT_FILE}")

# ── Main Function ─────────────────────────────────────────────────────────────

def data_quality_check(filepath=PROCESSED_FILE):
    """
    Runs all quality checks on the processed dataset.
    Call this function to run everything.
    """
    print("\n🔍 ReviewGuard — Data Quality Check Starting...\n")

    # Load the processed data
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("   Run data_collector.py first!")
        return

    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows from {filepath}")

    # Run all checks
    results = {}
    results["Missing Fields"]          = check_missing_fields(df)
    results["Duplicate Reviews"]       = check_duplicates(df)
    results["Suspicious Reviewers"]    = check_suspicious_reviewers(df)
    results["Rating Anomalies"]        = check_rating_anomalies(df)
    results["Date Validation"]         = check_date_validation(df)

    # Generate report
    generate_report(results)

    print("\n✅ Data Quality Check Complete!")
    print(f"   Report saved at: {REPORT_FILE}")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_quality_check()