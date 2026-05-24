🚀 Customer Churn Prediction App

This project leverages machine learning to predict customer churn, allowing
businesses to identify "at-risk" customers proactively. The model uses an SVC
(Support Vector Classifier) pipeline optimized with RFECV (Recursive Feature
Elimination with Cross-Validation) to ensure high performance and feature
efficiency.

🛠 Tech Stack

  - Machine Learning: scikit-learn, category_encoders, joblib
  - Web Framework: gradio
  - Data Processing: pandas, numpy
  - Infrastructure: Pipelines (Imputation, Scaling, OHE, Target Encoding)

📋 Features

  - Automated Preprocessing: The pipeline handles missing values, numerical
    scaling, One-Hot Encoding, and Target Encoding dynamically.
  - Feature Selection: Uses RFECV to automatically shortlist the most impactful
    features for prediction.
  - Confidence Scoring: Provides not just a prediction (Churn/Stay), but a
    probability percentage for model confidence.
  - Interactive UI: A clean, user-friendly interface built with Gradio.

📦 Project Structure

├── app.py                  # Main gradio application
├── requirements.txt        # Dependencies
├── customer_churn_model.pkl # The trained pipeline
├── shortlisted_features.pkl # List of features selected by RFECV
└── README.md

💡 How it Works

1.  Input: User fills out customer details in the Gradio interface.
2.  Pipeline: Data passes through the preprocessor (Imputation → Encoding →
    Scaling).
3.  Selection: The RFECV step filters the data to match the shortlisted
    features.
4.  Inference: The SVC model predicts the outcome and probability.
5.  Output: The app displays the result with a confidence percentage.

📈 Model Performance

The model was trained using a 5-fold Cross-Validation approach optimized for
ROC-AUC. It utilizes LinearSVC with L1 regularization for feature selection,
ensuring that the final model is both lightweight and accurate.

👤 Author

  - Saurav Kumar Yadav
