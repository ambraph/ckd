import streamlit as st
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Load the trained LightGBM model
model = load('lgbm_classifier_model.joblib')

# Define the input features
features = ['Age', 'Smoke', 'Diabetes', 'High_Cholesterol', 'BMI', 
            'FamilyHistoryCVD', 'SBP', 'DBP', 'Proteinuria']

# Page configuration
st.set_page_config(page_title='CVD Prediction App', layout='wide')

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["About", "Prediction", "Data Visualization", "Top Features Prediction", "Performance Metrics"])

# About Page
if page == "About":
    st.title("About This App")
    st.write("""
    This application is designed to predict the likelihood of cardiovascular disease (CVD) 
    based on several health metrics and personal information. 
    The model is built using a LightGBM classifier trained on relevant health data.
    """)

# Prediction Page
elif page == "Prediction":
    st.title("CVD Prediction Page")
    st.write("Please enter the following information:")

    # Input fields for the features
    user_input = {}
    for feature in features:
        if feature == 'Smoke':
            user_input[feature] = st.selectbox(feature, options=['Yes', 'No'])
        elif feature in ['Diabetes', 'High_Cholesterol', 'FamilyHistoryCVD', 'Proteinuria']:
            user_input[feature] = st.selectbox(feature, options=['Yes', 'No'])
        else:
            user_input[feature] = st.number_input(feature, min_value=0.0)

    # Convert user input into a DataFrame
    input_data = pd.DataFrame(user_input, index=[0])

    # Convert categorical responses to binary
    input_data['Smoke'] = np.where(input_data['Smoke'] == 'Yes', 1, 0)
    input_data['Diabetes'] = np.where(input_data['Diabetes'] == 'Yes', 1, 0)
    input_data['High_Cholesterol'] = np.where(input_data['High_Cholesterol'] == 'Yes', 1, 0)
    input_data['FamilyHistoryCVD'] = np.where(input_data['FamilyHistoryCVD'] == 'Yes', 1, 0)
    input_data['Proteinuria'] = np.where(input_data['Proteinuria'] == 'Yes', 1, 0)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success("The model predicts that the individual is likely to have CVD.")
        else:
            st.success("The model predicts that the individual is unlikely to have CVD.")

        st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("Exploratory Data Analysis of the dataset.")

    # Load dataset
    df = pd.read_csv("master_dataset.csv")  

    # Correlation heatmap
    st.subheader("Heatmap of Feature Correlation")
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    # Create a figure for the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    # Display the heatmap using Matplotlib's imshow
    cax = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    # Add color bar
    fig.colorbar(cax)
    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    # Annotate the heatmap with correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')
    # Display the plot in Streamlit
    st.pyplot(fig)

    


    # Bar chart for categorical features
    st.subheader("Distribution of Categorical Features")
    # List of categorical features
    categorical_features = ['Smoke', 'Diabetes', 'High_Cholesterol', 'FamilyHistoryCVD', 'Proteinuria']
    # Iterate over each categorical feature and create a bar plot
    for feature in categorical_features:
        fig, ax = plt.subplots(figsize=(6, 4))        
        # Count the occurrences of each category (Yes/No or 0/1)
        value_counts = df[feature].value_counts()        # Plot the bar chart
        ax.bar(value_counts.index.astype(str), value_counts.values, color=['skyblue', 'orange'])
        # Set plot labels and title
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {feature}")
        # Display the plot in Streamlit
        st.pyplot(fig)


# Top Features Prediction Page
elif page == "Top Features Prediction":
    st.title("Top Features Prediction")

    # Extract top 10 important features
    top_features = ['Age', 'SBP', 'BMI', 'DBP', 'High_Cholesterol', 'Smoke', 'FamilyHistoryCVD', 'Diabetes', 'Proteinuria']

    user_input = {}
    for feature in top_features:
        if feature == 'Smoke':
            user_input[feature] = st.selectbox(feature, options=['Yes', 'No'])
        elif feature in ['Diabetes', 'High_Cholesterol', 'FamilyHistoryCVD', 'Proteinuria']:
            user_input[feature] = st.selectbox(feature, options=['Yes', 'No'])
        else:
            user_input[feature] = st.number_input(feature, min_value=0.0)

    input_data = pd.DataFrame(user_input, index=[0])
    input_data['Smoke'] = np.where(input_data['Smoke'] == 'Yes', 1, 0)
    input_data['Diabetes'] = np.where(input_data['Diabetes'] == 'Yes', 1, 0)
    input_data['High_Cholesterol'] = np.where(input_data['High_Cholesterol'] == 'Yes', 1, 0)
    input_data['FamilyHistoryCVD'] = np.where(input_data['FamilyHistoryCVD'] == 'Yes', 1, 0)
    input_data['Proteinuria'] = np.where(input_data['Proteinuria'] == 'Yes', 1, 0)

    if st.button("Predict with Top Features"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success("High risk of CVD based on top features.")
        else:
            st.success("Low risk of CVD based on top features.")

# Performance Metrics Page
elif page == "Performance Metrics":
    st.title("Model Performance Metrics")

    # Example metrics (replace with your test set predictions)
    y_test = pd.read_csv("y_test.csv")  # Replace with actual test labels
    y_pred = model.predict(pd.read_csv("X_test.csv"))  # Replace with actual test data

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("**Accuracy**:", accuracy)
    st.write("**Precision**:", precision)
    st.write("**Recall**:", recall)
    st.write("**F1 Score**:", f1)

    # Confusion Matrix
    # st.subheader("Confusion Matrix")
    # cm = confusion_matrix(y_test, y_pred)
    # plt.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    # st.pyplot()

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(pd.read_csv("X_test.csv"))[:, 1])
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    st.pyplot()

# Run the app
if __name__ == "__main__":
    st.write("Developed by Mr. Raphael Enihe")