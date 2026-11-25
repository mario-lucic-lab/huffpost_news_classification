# HuffPost News Category Classification

A machine learning and NLP project focused on multi-class text classification using the **HuffPost News Category Dataset** (42 news categories). The notebook covers exploratory data analysis (EDA), text preprocessing, feature engineering, and a comparative evaluation of multiple classification approaches.

## Dataset
* **Source:** [News Category Dataset (Kaggle / HuffPost)](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
* **Size:** ~200,000 news articles containing headlines, short descriptions, authors, dates, and category labels.
* *Note: The raw JSON dataset is excluded from this repository due to file size limits. Download it from Kaggle and place it in the working directory (or Google Drive if running in Colab) before executing the notebook.*

## Project Workflow
1. **Exploratory Data Analysis (EDA):** Class distribution analysis, missing value inspection, and text length statistics.
2. **Text Preprocessing:** Lowercasing, punctuation/noise removal, stopword filtering, and lemmatization using `spaCy`.
3. **Feature Engineering:** Combining headline and description text representations using **TF-IDF** and **Word2Vec** embeddings.
4. **Modeling & Evaluation:**
   * **Logistic Regression** (standard and class-balanced)
   * **Multi-Layer Perceptron (MLP)** neural network built with TensorFlow/Keras
   * **XGBoost** classifier
   * Unsupervised and reinforcement learning experiments

## Key Results
* Both **Logistic Regression** and the **MLP Neural Network** achieved **~62% overall accuracy** across 42 imbalanced classes, reaching F1-scores of **0.78–0.79** on dominant categories such as `POLITICS`, `WELLNESS`, and `STYLE & BEAUTY`.
* Applying class balancing (`class_weight='balanced'`) improved macro recall from **0.44 to 0.53** on minority classes at the cost of lower overall accuracy (**55%**).

## Technologies Used
* **Python 3**
* **Data & NLP:** `pandas`, `numpy`, `spaCy`, `gensim`, `scikit-learn`
* **Deep Learning & Boosting:** `TensorFlow / Keras`, `XGBoost`
* **Visualization:** `matplotlib`, `seaborn`
