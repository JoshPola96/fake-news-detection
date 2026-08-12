# Fake News Detection System

> **Scope** · Timeboxed technical assessment (short). Built to a brief under a fixed clock — scope decisions were deliberate.

> [!NOTE]
> **Built 2025. The ecosystem has moved since.**
> Dependencies here are unpinned, so a clean `pip install` today resolves to
> versions that did not exist when this was written and pandas 3.0, numpy 2.5, pytest 9 and black 26 have all landed since. Expect install or
> runtime breakage on a fresh environment. What is on offer is the engineering
> approach and the decisions behind it, not a guaranteed-green build.
> Happy to bring it current if that would be useful — just ask.

## Overview
This project implements a machine learning pipeline to classify news articles as "Real" or "Fake" using advanced NLP techniques and optimized machine learning models. The system achieves high accuracy through careful preprocessing, feature engineering, and model optimization.

## Key Features
- **Text Preprocessing Pipeline**: Cleans and normalizes text data using NLTK
- **Advanced Feature Extraction**: Utilizes optimized TF-IDF with n-grams
- **Model Optimization**: Bayesian hyperparameter tuning for XGBoost and MLP classifiers
- **Visual Analytics**: Includes word clouds and distribution visualizations
- **Deployment Ready**: Streamlit web interface for real-time predictions

## Technical Skills Demonstrated
### Natural Language Processing
- Text cleaning (HTML removal, punctuation handling)
- Tokenization and lemmatization
- Stopword removal
- N-gram feature extraction
- TF-IDF vectorization optimization

### Machine Learning
- Hyperparameter optimization using Bayesian search
- Neural Networks (MLP) implementation
- Gradient Boosting (XGBoost) implementation
- Model evaluation metrics (precision, recall, F1-score)
- Confusion matrix visualization

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detector.git
```

2. Install dependencies:

3. Download NLTK resources:
```python
import nltk
nltk.download(['wordnet', 'stopwords', 'punkt', 'omw-1.4'])
```

## Usage
1. Preprocess the data:
```python
python preprocessing.py
```

2. Train and evaluate models:
```python
python model_training.py
```

3. Run the web app:
```python
streamlit run app.py
```
