import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Step 1: Load the Data
data = pd.read_csv('C:/Users/godas/OneDrive/Desktop/death_prediction_project/data.csv')

# Step 2: Data Cleaning and Feature Engineering
data['start_date'] = pd.to_datetime(data['start_date'], errors='coerce')
data['end_date'] = pd.to_datetime(data['end_date'], errors='coerce')
data = data.dropna()

# Define the target variable (1 = death occurred, 0 = no death)
data['target'] = (data['total_deaths'] > 0).astype(int)

# Define features and target variable
X = data[['population', 'covid_deaths', 'expected_deaths', 'excess_deaths']]
y = data['target']

# Check the original class distribution
print("Original class distribution in the data:", np.bincount(y))

# Proceed only if both classes are present
if len(np.unique(y)) < 2:
    print("Error: Only one class is present in the data. Consider collecting more data or using an anomaly detection approach.")
else:
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training class distribution before SMOTE:", np.bincount(y_train))

    # Apply SMOTE to balance the classes in the training set if needed
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Training class distribution after SMOTE:", np.bincount(y_train))

    # Step 3: Data Preprocessing using ColumnTransformer and Pipeline
    numeric_features = ['population', 'covid_deaths', 'expected_deaths', 'excess_deaths']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Step 4: Define and Train Models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Machine': SVC()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {score:.2f}')
        if score > best_score:
            best_score = score
            best_model = clf

    # Step 5: Save the best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print("Best model saved as 'best_model.pkl'")
