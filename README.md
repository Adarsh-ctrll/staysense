# 🏠 StaySense — NYC Airbnb Room Type Predictor

<p align="center">
  <b>Predict the room type of an NYC Airbnb listing using Machine Learning.</b>
</p>

<p align="center">
  <a href="https://staysense-1.onrender.com/">🌐 Live Demo</a>
  &nbsp; • &nbsp;
  <a href="https://staysense-xh91.onrender.com/docs">📚 API Docs</a>
  &nbsp; • &nbsp;
  <a href="https://github.com/Adarsh-ctrll/staysense/">💻 GitHub</a>
</p>

---

## 📌 Overview

**StaySense** is an end-to-end machine learning application built using NYC Airbnb listing data.

The application predicts whether an Airbnb listing is most likely to be:

- 🏠 **Entire Home/Apt**
- 🚪 **Private Room**
- 🛏️ **Shared Room**

The project covers the complete machine learning workflow — from data exploration and preprocessing to class-imbalance handling, model selection, hyperparameter tuning, evaluation, API development, and deployment.

---

## 🎯 Problem Statement

Airbnb listings differ significantly in terms of:

- Pricing
- Location
- Availability
- Reviews
- Host activity
- Minimum stay requirements

The objective of StaySense is to build a **multiclass classification model** that predicts the room type of an Airbnb listing based on its listing attributes.

The project also focuses on evaluating the model appropriately for an imbalanced classification problem rather than relying only on overall accuracy.

---

# ✨ Features

- 📊 NYC Airbnb data analysis
- 🤖 Multiclass room-type classification
- 🔢 Numerical and categorical feature preprocessing
- ⚖️ Class-imbalance analysis
- 🔍 SMOTE and SMOTENC experimentation
- 🌲 Random Forest classification
- 🎯 Hyperparameter tuning using `RandomizedSearchCV`
- 📈 Macro F1-based model evaluation
- 📊 Class-level performance analysis
- 🔌 FastAPI REST API
- 🖥️ Interactive web interface
- 📊 Prediction probabilities for all room types
- 🌐 Publicly deployed application

---

# 🧠 Machine Learning Workflow

```text
                    NYC Airbnb Dataset
                           │
                           ▼
                Data Exploration & Analysis
                           │
                           ▼
                 Data Cleaning & Preparation
                           │
                           ▼
                  Feature Preprocessing
                     ┌─────┴─────┐
                     │           │
                Numerical     Categorical
                  Features      Features
                     │           │
                     └─────┬─────┘
                           │
                           ▼
                 Class Imbalance Analysis
                     ┌─────┼─────┐
                     │     │     │
                   SMOTE SMOTENC Class Weighting
                     │     │     │
                     └─────┬─────┘
                           │
                           ▼
                 Random Forest Classifier
                           │
                           ▼
                 Hyperparameter Tuning
                    RandomizedSearchCV
                           │
                           ▼
                    Model Evaluation
                           │
                           ▼
                   Saved ML Pipeline
                           │
                           ▼
                    FastAPI REST API
                           │
                           ▼
                Interactive Web Interface
                           │
                           ▼
                 Room Type Prediction
                    + Probabilities
```
---

## 📊 Input Features

The model uses the following features to make predictions:

| Feature | Description |
|---|---|
| `latitude` | Geographic latitude of the listing |
| `longitude` | Geographic longitude of the listing |
| `price` | Price per night |
| `minimum_nights` | Minimum number of nights required |
| `number_of_reviews` | Total number of reviews |
| `reviews_per_month` | Average number of reviews per month |
| `calculated_host_listings_count` | Number of listings managed by the host |
| `availability_365` | Number of days available during the year |
| `neighbourhood_group` | NYC borough |
| `neighbourhood` | Specific neighbourhood |

## ⚙️ Model Development

The project uses a Scikit-learn Pipeline to ensure that preprocessing and model inference remain consistent between training and deployment.

### Preprocessing

The pipeline handles:

- Missing-value imputation
- Numerical feature transformation
- Feature scaling
- Categorical feature encoding

Numerical and categorical features are processed separately using a `ColumnTransformer`.

### Class Imbalance

The dataset contains an imbalanced distribution of room types.

Different approaches were evaluated, including:

- SMOTE
- SMOTENC
- `class_weight="balanced"`

The final Random Forest model uses class weighting to give greater importance to underrepresented classes.

## 🎯 Hyperparameter Tuning

RandomizedSearchCV was used to optimize the Random Forest classifier.

The search explored parameters including:

- `n_estimators`
- `max_depth`
- `min_samples_split`

The model was evaluated using 3-fold cross-validation with **Macro F1** as the primary scoring metric.

Macro F1 was selected because it gives equal importance to each room-type class and is more informative than accuracy when dealing with class imbalance.

## 📈 Model Performance

The final model achieved the following performance on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | **85.23%** |
| Macro F1 | **73.49%** |

The model was additionally evaluated using class-level precision, recall, F1-score, and a confusion matrix to understand performance across individual room types.

## 🚀 Deployment Architecture

StaySense uses a separate frontend and backend deployment.

```text
                    User
                     │
                     ▼
          ┌─────────────────────┐
          │    Render Static    │
          │       Frontend      │
          │  HTML / CSS / JS    │
          └──────────┬──────────┘
                     │
                 HTTP Request
                     │
                     ▼
          ┌─────────────────────┐
          │   Render Web        │
          │      Service        │
          │      FastAPI        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Saved ML Pipeline  │
          │  Random Forest      │
          └──────────┬──────────┘
                     │
                     ▼
              Prediction +
              Probabilities
```
