import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import firebase_admin
from firebase_admin import credentials, auth
from sklearn.model_selection import train_test_split

# Initializing Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("cvdpredictor-4452f-firebase-adminsdk-fl22e-228ccf5239.json")
    firebase_admin.initialize_app(cred)

# Loading the saved LightGBM model
with open('best_lightgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

df = pd.read_csv('balanced_feature_selected.csv')

# Authenticating functions
def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user
    except Exception as e:
        st.error("Login failed. Please check your email or password.")
        return None

def signup_user(email, password, display_name):
    try:
        user = auth.create_user(email=email, password=password, display_name=display_name)
        st.success("User created successfully! You can now log in.")
        return user
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return None

# Session state to track login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Main Page
if not st.session_state['logged_in']:
    st.title("Welcome to CVD Predictor App")
    option = st.radio("Choose an option:", ["Login", "Signup"])

    if option == "Login":
        st.title("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.success(f"Welcome {user.display_name}!")
                st.session_state['logged_in'] = True

    elif option == "Signup":
        st.title("Signup")
        display_name = st.text_input("Display Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Signup"):
            user = signup_user(email, password, display_name)
            if user:
                st.success("Signup successful! You can now log in.")

# Main Application
if st.session_state['logged_in']:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["About", "Prediction", "Model Evaluation", "Logout"])

    # Logout
    if page == "Logout":
        st.session_state['logged_in'] = False
        st.sidebar.success("You have been logged out.")

    # About Page
    if page == "About":
        st.title("About This App")
        st.write("""
        This application is a Cardiovascular Disease (CVD) Prediction Tool designed to help individuals assess their risk of developing cardiovascular disease.
        """)

    # Prediction Page
    elif page == "Prediction":
        st.title("CVD Prediction Page")
        age = st.number_input("Age", min_value=0)
        smoke = st.selectbox("Smoke", options=['Yes', 'No'])
        diabetes = st.selectbox("Diabetes", options=['Yes', 'No'])
        high_cholesterol = st.selectbox("High Cholesterol", options=['Yes', 'No'])
        bmi = st.number_input("BMI", min_value=0.0)
        family_history_cvd = st.selectbox("Family History of CVD", options=['Yes', 'No'])
        sbp = st.number_input("Systolic Blood Pressure (SBP)", min_value=0)
        dbp = st.number_input("Diastolic Blood Pressure (DBP)", min_value=0)
        proteinuria = st.selectbox("Proteinuria", options=['Yes', 'No'])

        user_data = pd.DataFrame({
            'Age': [age],
            'Smoke': [1 if smoke == 'Yes' else 0],
            'Diabetes': [1 if diabetes == 'Yes' else 0],
            'High_Cholesterol': [1 if high_cholesterol == 'Yes' else 0],
            'BMI': [bmi],
            'FamilyHistoryCVD': [1 if family_history_cvd == 'Yes' else 0],
            'SBP': [sbp],
            'DBP': [dbp],
            'Proteinuria': [1 if proteinuria == 'Yes' else 0]
        })

        if st.button("Predict"):
            prediction = model.predict(user_data)
            prediction_proba = model.predict_proba(user_data)[:, 1]

            if prediction[0] == 1:
                st.success("The model predicts that the individual is likely to have CVD.")
            else:
                st.success("The model predicts that the individual is unlikely to have CVD.")

            st.write(f"Probability of having CVD: {prediction_proba[0]:.2f}")

    # Model Evaluation Page
    elif page == "Model Evaluation":
        st.title("Model Evaluation")
        X = df.drop(columns=['Cvd'])
        y = df['Cvd']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        st.write(f"**Accuracy Score:** {accuracy:.2f}")
        st.write(f"**ROC AUC Score:** {roc_auc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix Plot")
       # Display Confusion Matrix as an Image
        st.subheader("Confusion Matrix")
        st.image("LightGBM Confusion Matrix.png", caption="Confusion Matrix", use_column_width=True)



        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig)
