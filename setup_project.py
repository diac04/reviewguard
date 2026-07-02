import os

print("Running setup_project.py...")
print("Current working directory:", os.getcwd())
print("-" * 40)

folders = [
    "data/raw",
    "data/processed",
    "data/graphs",
    "notebooks",
    "src",
    "outputs/charts",
    "outputs/reports"
]

files = [
    "README.md",
    "requirements.txt",
    "src/__init__.py",
    "src/data_collector.py",
    "src/data_quality.py",
    "src/feature_engineering.py",
    "src/graph_builder.py",
    "src/trust_score.py",
    ".gitignore"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print("Created folder:", folder)

# Create files
for file in files:
    folder_name = os.path.dirname(file)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        f.write("")
    print("Created file:", file)

print("-" * 40)
print("✅ Project structure created successfully!")