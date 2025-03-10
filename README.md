
# CVD Predictor App

The **CVD Predictor App** is a user-friendly tool designed to assess the likelihood of cardiovascular disease (CVD) based on user input and a machine learning model. This application leverages a LightGBM model and integrates Firebase for user authentication, allowing users to log in and make predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Firebase Setup](#firebase-setup)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model](#model)
- [App Pages](#app-pages)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **CVD Predictor App** aims to provide an accessible way for individuals to estimate their risk of developing cardiovascular disease. It uses a machine learning model trained on a balanced dataset with selected features. The app also includes model evaluation metrics and visualizations to help users understand the model's performance.

## Features

- **User Authentication**: Users can sign up and log in using Firebase Authentication.
- **CVD Prediction**: Based on user inputs, the app predicts the likelihood of cardiovascular disease.
- **Model Evaluation**: The app provides metrics such as accuracy, ROC AUC score, confusion matrix, and ROC curve for model performance evaluation.
- **Interactive Interface**: Built using Streamlit for an interactive and responsive user experience.

## Installation

To get started with the app, follow the instructions below.

### Prerequisites

- Python 3.8 or higher
- Firebase Admin SDK credentials file (`cvdpredictor-4452f-firebase-adminsdk-fl22e-228ccf5239.json`)
- The following Python packages:
  - `streamlit`
  - `pickle`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `firebase-admin`
  - `lightgbm`

### Step 1: Clone the Repository

```bash
git clone https://github.com/pathfinderNdoma/cvd-predictor-app.git
cd cvd-predictor-app
```

### Step 2: Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Firebase Setup

Ensure you have the Firebase Admin SDK credentials file (`cvdpredictor-4452f-firebase-adminsdk-fl22e-228ccf5239.json`) in the root directory of your project.

### Step 4: Run the App

Start the Streamlit app using the following command:

```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Firebase Setup

This app uses Firebase for user authentication. Follow these steps to set up Firebase:

1. Go to [Firebase Console](https://console.firebase.google.com/).
2. Create a new project and navigate to **Project Settings**.
3. Under the **Service Accounts** tab, generate a new private key. Download the JSON file and place it in the root directory of your project.
4. Replace `cvdpredictor-4452f-firebase-adminsdk-fl22e-228ccf5239.json` with your downloaded JSON file.

## Usage

1. Launch the app using the command: `streamlit run app.py`.
2. Choose to either **Login** or **Signup**.
3. If signing up, provide your display name, email, and password.
4. Once logged in, navigate through the sidebar options:
   - **About**: Learn more about the app.
   - **Prediction**: Enter your health data to get a CVD risk prediction.
   - **Model Evaluation**: View metrics and visualizations of the model's performance.
   - **Logout**: Log out of your session.

## Data Description

The dataset used for this model contains the following features:

| Feature                | Description                          |
|------------------------|--------------------------------------|
| `Age`                  | Age of the individual                |
| `Smoke`                | Smoking status (Yes/No)              |
| `Diabetes`             | Presence of diabetes (Yes/No)        |
| `High_Cholesterol`     | High cholesterol status (Yes/No)     |
| `BMI`                  | Body Mass Index                      |
| `FamilyHistoryCVD`     | Family history of CVD (Yes/No)       |
| `SBP`                  | Systolic Blood Pressure              |
| `DBP`                  | Diastolic Blood Pressure             |
| `Proteinuria`          | Presence of protein in urine (Yes/No)|
| `Cvd`                  | Target variable (1: CVD, 0: No CVD)  |

The dataset is preprocessed and split into features (`X`) and the target variable (`y`).

## Model

The prediction model is built using **LightGBM**, a gradient boosting framework. The model was trained on a balanced and feature-selected dataset. The saved model (`best_lightgbm_model.pkl`) is loaded into the app for making predictions.

### Model Evaluation

The following evaluation metrics are provided:

- **Accuracy Score**: The proportion of correct predictions.
- **ROC AUC Score**: The area under the ROC curve, indicating model performance.
- **Confusion Matrix**: A matrix displaying the true positive, true negative, false positive, and false negative counts.
- **ROC Curve**: A plot showing the true positive rate against the false positive rate.

## App Pages

The app has four main pages:

1. **About**: Provides an overview of the app's purpose and functionality.
2. **Prediction**: Allows users to input their data and receive a prediction on their CVD risk.
3. **Model Evaluation**: Displays the model's performance metrics and visualizations, including the confusion matrix and ROC curve.
4. **Logout**: Logs the user out of their session.

## Contributing

Contributions are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

```
# CVD Predictor app
