import pandas as pd

# Check Mendeley dataset
print("=== Mendeley Dataset ===")
df1 = pd.read_csv("data/Mendeley_dataset.csv", encoding="latin-1", on_bad_lines='skip')
df1.columns = df1.columns.str.strip()
print("Columns:", df1.columns.tolist())
print("Shape:", df1.shape)
print(df1.head(2))

print("\n=== Student Dataset ===")
df2 = pd.read_csv("data/Student_dataset.csv", encoding="latin-1", on_bad_lines='skip')
df2.columns = df2.columns.str.strip()
print("Columns:", df2.columns.tolist())
print("Shape:", df2.shape)
print(df2.head(2))