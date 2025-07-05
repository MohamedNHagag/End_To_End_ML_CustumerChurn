# 📊 Customer Churn Prediction - End-to-End Machine Learning Project

This is an **End-to-End Machine Learning Project** for predicting customer churn using various classification algorithms such as:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- CatBoost

The project includes:
- Data ingestion
- Data transformation (encoding, scaling)
- Model training with hyperparameter tuning
- Model evaluation
- Model saving & loading
- Prediction pipeline

---

## 📂 **Project Structure**
```
Customer_Churn_Prediction_END_TO_END_ML/
│
├── artifacts/                     # Stores intermediate files like models, preprocessors, datasets
│
├── src/
│   ├── components/
│   │   ├── ingestion.py           # Data ingestion module
│   │   ├── transformation.py      # Data preprocessing module
│   │   ├── trainer.py             # Model training module
│   │   ├── evaluate.py            # Model evaluation module
│   │
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   ├── utils.py                   # Utility functions (saving & loading objects)
│   │
│   └── pipeline/
│       └── prediction.py          # Prediction pipeline
│
├── main.py                        # Main training script
│
└── README.md                      # Project documentation (this file)
```

---

## 📥 **How to Run the Project**

### 1️⃣ Install Requirements:
```bash
pip install -r requirements.txt
```


### 2️⃣ Run Main Training Pipeline:
```bash
python main.py
```

---

## ✅ **Workflow**
1. Reads raw dataset (CSV)
2. Splits dataset into **Train** & **Test**
3. Encodes categorical variables & scales numerical columns
4. Trains multiple classification models
5. Evaluates models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
6. Selects **best performing model**
7. Saves model for future use