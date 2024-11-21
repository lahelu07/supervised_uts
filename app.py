import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model function
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Prediction function
def predict_gender(model, features):
    input_data = pd.DataFrame([features])
    input_data = input_data[model.feature_names_in_]  # Ensure column order matches training data
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Gender Prediction with Model Selection")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model for Prediction:",
        ["Random Forest", "Decision Tree", "Logistic Regression"]
    )
    model_file_mapping = {
        "Random Forest": "random_forest_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl"
    }

    # Load selected model
    model_file = model_file_mapping[model_choice]
    model = load_model(model_file)

    # Input features from the user
    st.subheader("Enter Features:")
    forehead_width_cm = st.number_input("Forehead Width (cm):", min_value=0.0, step=0.1)
    forehead_height_cm = st.number_input("Forehead Height (cm):", min_value=0.0, step=0.1)
    nose_wide = st.selectbox("Nose Wide (0 = No, 1 = Yes):", [0, 1])
    nose_long = st.selectbox("Nose Long (0 = No, 1 = Yes):", [0, 1])
    lips_thin = st.selectbox("Lips Thin (0 = No, 1 = Yes):", [0, 1])
    distance_nose_to_lip_long = st.selectbox("Distance Nose to Lip Long (0 = No, 1 = Yes):", [0, 1])
    long_hair = st.selectbox("Long Hair (0 = No, 1 = Yes):", [0, 1])

    # Prepare features for prediction
    features = {
        "forehead_width_cm": forehead_width_cm,
        "forehead_height_cm": forehead_height_cm,
        "nose_wide": nose_wide,
        "nose_long": nose_long,
        "lips_thin": lips_thin,
        "distance_nose_to_lip_long": distance_nose_to_lip_long,
        "long_hair": long_hair
    }

    # Button to make predictions and display confusion matrix
    if st.button("Predict and Show Confusion Matrix"):
        prediction = predict_gender(model, features)
        predicted_gender = "Male" if prediction == 1 else "Female"
        st.subheader(f"Predicted Gender: {predicted_gender}")

        # Simulate true labels and predictions for confusion matrix (example data)
        true_labels = [0, 1, 1, 0, 1, 0]  # Replace with actual labels if available
        sample_predictions = [predict_gender(model, features) for _ in true_labels]

        # Calculate and display confusion matrix
        cm = confusion_matrix(true_labels, sample_predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"])

        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
