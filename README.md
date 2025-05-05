# Resume-Categorization-Using-NLP


This project focuses on automating the classification of resumes into predefined job categories using Natural Language Processing (NLP) techniques and machine learning models. The aim is to streamline candidate screening in recruitment workflows.

## 🚀 Project Overview

Recruiters often need to sift through hundreds of resumes to identify candidates for specific job roles. This project leverages NLP and supervised learning techniques to automatically categorize resumes based on their content into categories such as Data Scientist, Web Developer, HR, etc.

## 📁 Dataset

- Format: Collection of `.txt` or `.pdf` resume files labeled by job role.
- Categories: Data Scientist, Web Developer, HR, Software Engineer, etc.

> *Note: Due to privacy concerns, actual resumes are anonymized or synthetically generated.*

## 🔍 Key Features

- Text preprocessing (tokenization, stop word removal, lemmatization)
- Feature extraction using TF-IDF and/or word embeddings (Word2Vec, BERT)
- Multi-class classification using models like:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors
  - Naive Bayes
  - Support Vector Classifier (SVC)
  - One-vs-Rest Classifier
- Model evaluation with accuracy, precision, recall, and F1-score
- Interactive Streamlit web app for real-time resume prediction

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: NLTK, spaCy, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Modeling**: Logistic Regression, SVC, Random Forest, Naive Bayes, KNN
- **Deployment**: Streamlit (for web interface)


## 📊 Results

The following models were evaluated on the test dataset:

| Model                            | Accuracy (%) |
|----------------------------------|--------------|
| Logistic Regression              | 99.48        |
| Support Vector Classifier (SVC)  | 99.48        |
| Random Forest Classifier         | 98.96        |
| K-Nearest Neighbors (KNN)        | 98.45        |
| One-vs-Rest Classifier           | 98.45        |
| Multinomial Naive Bayes          | 97.93        |


## 🚧 Future Improvements

- Incorporate transformer-based embeddings (e.g., BERT, RoBERTa)
- PDF/Docx resume parsing
- Real-time feedback for resume improvement
- ATS (Applicant Tracking System) integration

