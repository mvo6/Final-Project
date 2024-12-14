import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Function to clean social media usage data
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Load and clean the data
@st.cache_data
def load_data():
    df = pd.read_csv("social_media_usage.csv")
    df['sm_li'] = clean_sm(df['web1h'])
    df['income'] = np.where(df['income'] <= 9, df['income'], np.nan)
    df['education'] = np.where(df['educ2'] <= 8, df['educ2'], np.nan)
    df['age'] = np.where(df['age'] <= 98, df['age'], np.nan)
    df['parent'] = clean_sm(df['par'])
    df['married'] = clean_sm(df['marital'])
    df['female'] = clean_sm(df['gender'] == 1)
    return df[['income', 'education', 'parent', 'married', 'female', 'age', 'sm_li']].dropna()

# Load the data
data = load_data()

# Split data into training and testing
X = data.drop(columns=['sm_li'])
y = data['sm_li']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

# Streamlit app
st.title("LinkedIn User Predictor")
st.markdown("This app predicts whether someone is likely to use LinkedIn based on demographic features.")

# Input form for user features
st.header("Enter User Information:")
income = st.slider("Income (1-9):", 1, 9, value=5)
education = st.slider("Education (1-8):", 1, 8, value=4)
parent = st.selectbox("Parent (Yes=1, No=0):", [1, 0])
married = st.selectbox("Married (Yes=1, No=0):", [1, 0])
female = st.selectbox("Female (Yes=1, No=0):", [1, 0])
age = st.slider("Age:", 1, 98, value=30)

# Predict button
if st.button("Predict"):
    # Prepare input data for prediction
    user_input = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [parent],
        'married': [married],
        'female': [female],
        'age': [age]
    })
    
    # Make predictions
    probability = log_reg.predict_proba(user_input)[0][1]
    prediction = log_reg.predict(user_input)[0]
    
    # Display results
    st.subheader("Prediction Result")
    st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Not LinkedIn User'}")
    st.write(f"Probability of LinkedIn Usage: {probability:.2f}")

