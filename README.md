# 🎵 Music Genre Classifier

A web-based music genre classification system built with **Apache Spark MLlib** and **Flask**. Classifies song lyrics into 8 genres using a TF-IDF + Logistic Regression pipeline.
----
<img width="1535" height="1080" alt="UI" src="https://github.com/user-attachments/assets/9ee20754-0533-4943-9f0d-83b8832333b4" />


---

## 📁 Project Structure

```
music-classifier/
├── run.bat                   ← Start everything (double-click)
├── train.py                  ← Train model on 7 Mendeley genres
├── retrain.py                ← Retrain model on all 8 genres
├── Student_dataset.csv       ← 100 Soul songs (custom dataset)
├── Merged_dataset.csv        ← Mendeley + Student combined
├── app/
│   ├── server.py             ← Flask web server
│   ├── templates/
│   │   └── index.html        ← Web UI with bar chart
│   └── static/
├── data/
│   ├── Mendeley_dataset.csv
│   ├── Student_dataset.csv
│   └── Merged_dataset.csv
└── model/                    ← Saved Spark MLlib model (auto-generated)
    └── labels.txt
```

---

## 🎯 Genres Supported

| # | Genre    | Source          |
|---|----------|-----------------|
| 1 | Pop      | Mendeley        |
| 2 | Country  | Mendeley        |
| 3 | Blues    | Mendeley        |
| 4 | Jazz     | Mendeley        |
| 5 | Reggae   | Mendeley        |
| 6 | Rock     | Mendeley        |
| 7 | Hip Hop  | Mendeley        |
| 8 | Soul     | Student dataset |

---

## ⚙️ Requirements

- Python 3.10
- Java 11 (OpenJDK Temurin)
- PySpark 3.4.1
- Flask

Install dependencies:
```bash
pip install pyspark==3.4.1 flask
```

### Windows Setup (Required)

Download and place these files in `C:\hadoop\bin\`:
- [`winutils.exe`](https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe)
- [`hadoop.dll`](https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/hadoop.dll)

Then copy `hadoop.dll` to `C:\Windows\System32\`:
```bat
copy C:\hadoop\bin\hadoop.dll C:\Windows\System32\hadoop.dll
```

Set environment variable:
- `HADOOP_HOME` = `C:\hadoop`
- Add `C:\hadoop\bin` to `PATH`

---

## 🚀 How to Run

### Option 1 — Double click (easiest)
```
run.bat
```
This will:
1. Check if model exists, train if not
2. Start the Flask server
3. Open browser at `http://localhost:5000`

### Option 2 — Manual steps

**Step 1: Train the model (first time)**
```bash
python train.py
```

**Step 2: Retrain with all 8 genres**
```bash
python retrain.py
```

**Step 3: Start the server**
```bash
python app/server.py
```

**Step 4: Open browser**
```
http://localhost:5000
```

---

## 🧠 ML Pipeline

```
Raw Lyrics
    ↓
Tokenizer          (split into words)
    ↓
StopWordsRemover   (remove "the", "a", "is" etc.)
    ↓
HashingTF          (convert words to feature vectors, 20,000 features)
    ↓
IDF                (weight words by importance)
    ↓
LogisticRegression (classify into 8 genres)
    ↓
Predicted Genre + Confidence Scores
```

- Train/Test split: **80% / 20%**
- Model saved to: `model/` folder

---

## 🌐 Web UI

- Paste any song lyrics into the text box
- Click **Classify Genre**
- See a **horizontal bar chart** showing confidence % for all 8 genres
- The top predicted genre is highlighted

---

## 📊 Dataset Info

| Dataset         | Rows    | Genres |
|-----------------|---------|--------|
| Mendeley        | ~28,000 | 7      |
| Student (Soul)  | 100+    | 1      |
| Merged (total)  | ~28,100 | 8      |

Columns in all datasets:
```
artist_name, track_name, release_date, genre, lyrics
```

---


