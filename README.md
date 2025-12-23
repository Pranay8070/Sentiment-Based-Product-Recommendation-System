
---

# 🛍️ Sentiment-Based Product Recommendation System

## 📌 Overview

With the rapid growth of e-commerce platforms, personalized product recommendations have become essential to improve customer experience and drive business growth. Industry leaders like Amazon and Flipkart rely heavily on **machine learning, NLP, and recommender systems** to understand user preferences and feedback.

This project implements an **end-to-end Sentiment-Based Product Recommendation System** for a fictional e-commerce company **Ebuss**. The system intelligently combines **sentiment analysis of customer reviews** with **collaborative filtering** to recommend products that are both relevant and positively received by users.

---

## 🎯 Problem Statement

Ebuss operates across multiple product categories, including:

* Household essentials
* Books
* Personal care and beauty products
* Medicines and healthcare
* Electrical appliances
* Kitchen and dining products

To compete with established market leaders, Ebuss requires an intelligent system that:

* Analyzes customer sentiment from textual reviews
* Recommends products based on user similarity and preferences
* Filters recommendations using positive sentiment to improve quality

---

## ✅ Solution

The proposed solution includes:

* Sentiment classification using NLP and machine learning
* Collaborative filtering-based recommender system
* Hybrid recommendation logic combining sentiment and relevance
* Full-stack deployment using Flask and Heroku

🔗 **GitHub Repository**
(https://github.com/Pranay8070/Sentiment-Based-Product-Recommendation-System)

🔗 **Live Application (Heroku)**
(https://sentiment8070-562abb5d278e.herokuapp.com/)

---

## 🛠️ Tech Stack

### Programming & Frameworks

* Python 3.9.7
* Flask 2.0.2
* Bootstrap CDN 5.1.3

### Machine Learning & NLP

* scikit-learn 1.0.2
* XGBoost 1.5.1
* NumPy 1.22.0
* Pandas 1.3.5
* NLTK 3.6.7

---

## 📂 Project Structure

```
SentimentBasedProductRecommendation/
│
├── dataset/                          # Dataset and attribute description
├── pickle/                           # Saved ML models
├── templates/
│   └── index.html                    # Flask UI (Jinja + Bootstrap)
│
├── SentimentBasedProductRecommendation.ipynb
├── model.py                          # Sentiment prediction & filtering logic
├── app.py                            # Flask application
├── requirements.txt
└── README.md
```

---

## 🧠 Solution Approach

### 1️⃣ Data Cleaning & Preprocessing

* Removed missing and noisy data
* Performed exploratory data analysis
* Applied text preprocessing techniques:

  * Tokenization
  * Stopword removal
  * Lemmatization

---

### 2️⃣ Text Vectorization

* Combined `review_title` and `review_text`
* Applied **TF-IDF Vectorizer** to convert text into numerical features
* Captures the relative importance of words across documents

---

### 3️⃣ Handling Class Imbalance

* Dataset suffered from class imbalance in sentiment labels
* Applied **SMOTE (Synthetic Minority Oversampling Technique)** before model training

---

### 4️⃣ Sentiment Classification

Machine learning models evaluated:

* Logistic Regression
* Naive Bayes
* Decision Tree
* Random Forest
* **XGBoost**

**Evaluation Metrics**:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

✅ **Best Performing Model**: **XGBoost**

---

### 5️⃣ Recommender System

* Implemented **Collaborative Filtering**:

  * User–User similarity
  * Item–Item similarity

**Evaluation Metric**:

* RMSE (Root Mean Square Error)

The best-performing recommender model is used to generate **Top 20 product recommendations**.

---

### 6️⃣ Sentiment-Based Filtering

* Predicted sentiment for reviews associated with the top 20 recommended products
* Filtered and ranked products based on **positive user sentiment**
* Final output consists of **Top 5 products with highest positive sentiment**

---

### 7️⃣ Model Serving & Deployment

* Trained ML models saved as pickle files
* Flask API used to serve predictions
* UI built using Flask Jinja templates and Bootstrap
* Deployed as a complete end-to-end application on **Heroku**

---

## 🚀 Run the Project Locally

```bash
# Clone the repository
git clone https://github.com/Pranay8070/Sentiment-Based-Product-Recommendation-System.git

# Navigate to the project directory
cd SentimentBasedProductRecommendation

# Install required dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

Open your browser and visit:

```
http://127.0.0.0:5000/
```

---

## 📈 Key Takeaways

* Built a real-world NLP-based sentiment classifier
* Handled class imbalance using SMOTE
* Designed collaborative filtering recommender systems
* Combined sentiment analysis with recommendations
* Deployed a machine learning application end-to-end

---

