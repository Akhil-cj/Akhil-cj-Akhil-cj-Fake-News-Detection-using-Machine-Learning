
# Fake News Detection using Machine Learning

## 📚 Project Overview
This project builds a machine learning model to detect fake news using Natural Language Processing (NLP) techniques.  
By transforming news articles into TF-IDF feature vectors and training a **Passive Aggressive Classifier**, the system can effectively distinguish between **real** and **fake** news with high accuracy.  
This work addresses the growing problem of misinformation across digital platforms.

---

## 🎯 Objectives
- Build an efficient model to **classify news articles** as Real or Fake.
- Apply **TF-IDF vectorization** for feature extraction from text.
- Use the **Passive Aggressive Classifier** for effective and scalable fake news detection.
- Evaluate model performance using **accuracy score** and **confusion matrix**.

---

## 📊 Data Collection
- **Dataset**: CSV file (`news.csv`) containing labeled news articles (`REAL` or `FAKE`).
- **Fields**:
  - `text`: Full news article.
  - `label`: Ground truth (REAL/FAKE).
- **Source**: Typically from publicly available fake news datasets like Kaggle's **Fake News Detection Challenge**.

---

## 🛠️ Technologies Used
- **Python**
- **Libraries**:  
  - `pandas`, `numpy`
  - `scikit-learn` (TF-IDF Vectorizer, Passive Aggressive Classifier, evaluation metrics)
  - `matplotlib`, `seaborn` (for visualization)

---

## 🧹 Methodology
1. **Data Exploration**: Load and inspect the dataset using pandas.
2. **Preprocessing**: 
   - Separate text and labels.
   - Split into training and test sets (80%-20%).
3. **Feature Extraction**: 
   - Use `TfidfVectorizer` to convert text into numerical features.
   - Apply stop word removal and limit common word influence (`max_df=0.7`).
4. **Model Training**:
   - Train a **PassiveAggressiveClassifier** (`max_iter=50`) on the training data.
5. **Evaluation**:
   - Predict on test data.
   - Calculate **accuracy**.
   - Plot **confusion matrix** and generate **classification report**.

---

## 📈 Results
- **Accuracy Achieved**: ~92%
- **Confusion Matrix**:  
  Analyzed True Positives, True Negatives, False Positives, and False Negatives.
- **Classification Report**:  
  Precision, Recall, F1-Score calculated for both FAKE and REAL classes.

---

## 🚧 Challenges Faced
- **Handling noise** in text (stopwords, frequent words).
- **Model tuning** for best performance vs computational cost.
- **Managing class imbalance** between REAL and FAKE news articles.

---

## 🔍 Insights
- TF-IDF effectively captured important textual patterns.
- Passive Aggressive Classifier performed well for binary classification tasks.
- Further tuning can enhance detection precision and recall based on real-world needs.

---

## 🚀 Future Enhancements
- **Hyperparameter optimization** for better performance.
- **Handling imbalanced data** using techniques like SMOTE or undersampling.
- **Deploying as a web app** for real-time fake news detection.
- **Experimenting with deep learning models** (e.g., LSTM, BERT).

---

## 📂 Files Included
- `Fake News Detection Project using Machine Learning.py`: Full code for training, evaluating, and visualizing the model.
- `report_fake news.pdf`: Detailed project report explaining methodology, challenges, and findings.

---

## 📌 How to Run
1. Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
2. Place `news.csv` in the working directory.
3. Run the `Fake News Detection Project using Machine Learning.py` script.
4. Review accuracy, confusion matrix, and model evaluation outputs.

---

> **Project submitted by**: AKHIL C J

---
