import pandas as pd

csv_path = "data/Merged_dataset.csv"
clean_path = "data/Merged_dataset_clean.csv"

print("Reading raw bytes...")

# Read raw bytes and try to decode with errors='ignore' (drops bad chars)
with open(csv_path, "rb") as f:
    raw = f.read()

# Force decode - ignore any undecodable bytes
text = raw.decode("latin-1")  # latin-1 NEVER fails - it maps all 256 bytes

# Write as clean UTF-8
with open("data/temp_fixed.csv", "w", encoding="utf-8") as f:
    f.write(text)

print("Forced decode done. Now reading with pandas...")

# Now read the UTF-8 version
df = pd.read_csv(
    "data/temp_fixed.csv",
    on_bad_lines='skip',
    engine='python',
    quoting=3  # QUOTE_NONE
)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Fix column names - strip spaces and BOM
df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('"', '')
print("Cleaned columns:", df.columns.tolist())