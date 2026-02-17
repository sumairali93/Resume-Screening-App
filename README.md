# 📄 Resume Screening App

An end-to-end NLP-powered web application that automatically classifies resumes into **25 job categories** with **98.45% test accuracy**. Built with Scikit-learn and deployed as an interactive web app using Streamlit.

---

## 🚀 Live Demo

Upload a resume (PDF or TXT) → get instant job category prediction.

![Resume Screening App](https://via.placeholder.com/800x400?text=Resume+Screening+App+Demo)

---

## 📌 Problem Statement

HR teams manually screen hundreds of resumes per job posting — a slow, expensive, and inconsistent process. This app automates that initial screening step using NLP and machine learning, classifying any uploaded resume into the most relevant job category in seconds.

---

## 🤖 How It Works

```
Resume (PDF/TXT)
       ↓
  Text Extraction
       ↓
  Text Cleaning (regex-based NLP)
       ↓
  TF-IDF Vectorization (7351 features)
       ↓
  KNN Classifier (OneVsRest)
       ↓
  Predicted Job Category
```

---

## 🏷️ 25 Job Categories Supported

| | | | | |
|---|---|---|---|---|
| Data Science | Python Developer | Java Developer | DevOps Engineer | DotNet Developer |
| Testing | Automation Testing | Web Designing | HR | Operations Manager |
| Blockchain | ETL Developer | Hadoop | Database | Network Security Engineer |
| SAP Developer | Business Analyst | Mechanical Engineer | Electrical Engineering | Civil Engineer |
| PMO | Sales | Health and Fitness | Arts | Advocate |

---

## 📊 Dataset

- **File:** `UpdatedResumeDataSet.csv`
- **Size:** 962 resumes × 2 columns (`Category`, `Resume`)
- **Categories:** 25 job roles
- **Largest category:** Java Developer (84 resumes)

---

## ⚙️ ML Pipeline (Notebook)

### 1. Data Exploration
- Loaded 962 resume records across 25 categories
- Visualized category distribution with bar chart and pie chart

### 2. Text Cleaning (`cleanResume` function)
```python
- Remove URLs
- Remove RT/cc tokens
- Remove @mentions
- Remove hashtags
- Remove special characters/punctuation
- Remove non-ASCII characters
- Normalize whitespace
```

### 3. Label Encoding
- Converted 25 category strings → numeric labels using `LabelEncoder`

### 4. TF-IDF Vectorization
- `TfidfVectorizer(stop_words='english')`
- Produced **7,351 features** from resume text

### 5. Train/Test Split
- 80% train (769 samples) / 20% test (193 samples)
- `random_state=42`

### 6. Model Training
- **Algorithm:** K-Nearest Neighbors wrapped in `OneVsRestClassifier`
- Multi-class classification (25 classes)

### 7. Model Serialization
- Saved `tfidf.pkl` and `clf.pkl` using `pickle` for deployment

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **98.45%** |
| Training Accuracy | 98.57% |
| Mean CV Score (10-fold) | 96.88% |

> Near-identical train and test accuracy confirms **no overfitting**. 10-fold cross-validation scores ranged from 92.2% to 100%.

---

## 🌐 Streamlit Web App (`app.py`)

**Features:**
- Upload resume as **PDF** or **TXT**
- PDF text extraction using **PyMuPDF (fitz)**
- Handles both UTF-8 and Latin-1 encoded text files
- Real-time category prediction displayed instantly

**Key functions:**
```python
clean_resume(txt)          # NLP text cleaning pipeline
extract_text_from_pdf()    # PDF → text using PyMuPDF
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, KNN, LabelEncoder, cross-validation |
| Streamlit | Web app deployment |
| PyMuPDF (fitz) | PDF text extraction |
| Matplotlib / Seaborn | EDA visualizations |
| Pickle | Model serialization |

---

## 📁 Project Structure

```
Resume-Screening-App/
│
├── Resume_Screening_App.ipynb   # Full ML pipeline: EDA → training → evaluation
├── UpdatedResumeDataSet.csv     # 962 labeled resumes across 25 categories
├── app.py                       # Streamlit web application
├── clf.pkl                      # Trained KNN classifier (serialized)
├── tfidf.pkl                    # Fitted TF-IDF vectorizer (serialized)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sumairali93/Resume-Screening-App.git
cd Resume-Screening-App
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

Upload any resume in PDF or TXT format and get instant category prediction!

---

## ✅ Test It Yourself

The notebook includes a real resume test (Cell 30-31) — when Sumair Ali's own resume was fed into the model, it correctly predicted: **"Data Science"** ✓

---

## 💡 Key Learnings

- TF-IDF is highly effective for text classification — 7,351 features captured enough signal for 98%+ accuracy
- `OneVsRestClassifier` elegantly handles multi-class classification with binary classifiers
- Near-zero gap between train/test accuracy means the model generalizes well despite the small dataset
- PyMuPDF provides reliable PDF text extraction compared to alternatives like PyPDF2

---

## 👤 Author

**Sumair Ali**
- GitHub: [@sumairali93](https://github.com/sumairali93)
- LinkedIn: [linkedin.com/in/sumairali93](https://linkedin.com/in/sumairali93)
