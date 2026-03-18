import pandas as pd

# Load both
df1 = pd.read_csv("data/Mendeley_dataset.csv", encoding="latin-1", on_bad_lines='skip')
df2 = pd.read_csv("data/Student_dataset.csv", encoding="latin-1", on_bad_lines='skip')

# Strip column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

print("Mendeley columns:", df1.columns.tolist())
print("Student columns:", df2.columns.tolist())

# Keep only required columns
cols = ['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']
df1 = df1[cols]
df2 = df2[cols]

# Clean lyrics
for df in [df1, df2]:
    df['lyrics'] = df['lyrics'].astype(str)\
        .str.replace('\n', ' ')\
        .str.replace('\r', ' ')\
        .str.replace('"', "'")\
        .str.strip()

# Merge
merged = pd.concat([df1, df2], ignore_index=True)
merged = merged.dropna(subset=['lyrics', 'genre'])

print(f"\nMerged shape: {merged.shape}")
print("Genres:")
print(merged['genre'].value_counts())

# Save as proper UTF-8 CSV
merged.to_csv("data/Merged_dataset.csv", index=False, encoding="utf-8")
print("\n✅ Merged_dataset.csv recreated successfully!")