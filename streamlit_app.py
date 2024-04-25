import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
dataset = pd.read_csv('Data.csv')
dataset.columns = dataset.columns.str.strip()

# Function to load models
def load_models():
    return {
        'male': {
            'home': pickle.load(open('male_logistic_regression_model.pkl', 'rb')),
            'clinical1': pickle.load(open('male_random_forest_model.pkl', 'rb')),
            'clinical2': pickle.load(open('male_gradient_boosting_model.pkl', 'rb'))
        },
        'female': {
            'home': pickle.load(open('female_logistic_regression_model.pkl', 'rb')),
            'clinical1': pickle.load(open('female_random_forest_model.pkl', 'rb')),
            'clinical2': pickle.load(open('female_gradient_boosting_model.pkl', 'rb'))
        }
    }

models = load_models()

# Initialize LabelEncoders for categorical features
label_encoders = {
    'Exercise': LabelEncoder(),
    'Education': LabelEncoder()
}

def main():
    st.title("Sarcopenia Prediction")

    with st.form("prediction_form"):
        gender = st.selectbox('Select Gender', options=['male', 'female'])
        prediction_type = st.selectbox('Select Prediction Type', options=['home', 'clinical1', 'clinical2'])

        # Define input fields with options
        required_fields = {
            'home': ['Age', 'Weight', 'Height', 'Exercise', 'DM', 'Smoking', 'HT', 'Education', 'BMI', 'Smoking (packet/year)', 'Alcohol'],
            'clinical1': ['Age', 'Weight', 'Height', 'Exercise', 'DM', 'Smoking', 'HT', 'Education', 'BMI', 'Smoking (packet/year)', 'Alcohol', 'CST', 'Gait speed'],
            'clinical2': ['Age', 'Weight', 'Height', 'Exercise', 'DM', 'Smoking', 'HT', 'Education', 'BMI', 'Smoking (packet/year)', 'Alcohol', 'CST', 'Gait speed', 'Grip strength']
        }
        data = {}
        for field in required_fields[prediction_type]:
            if field == 'Exercise':
                options = ['Light', 'Moderate', 'High']
                data[field] = [st.selectbox(f'{field}', options)]
            elif field == 'Education':
                options = ['Primary School', 'Middle School', 'High School', 'University']
                data[field] = [st.selectbox(f'{field}', options)]
            elif field in ['DM', 'Smoking', 'HT']:
                data[field] = [1 if st.checkbox(f'{field}') else 0]
            else:
                data[field] = [st.number_input(f'{field}', format="%.2f")]

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        features = pd.DataFrame(data)
        # Encode categorical data
        for field in ['Exercise', 'Education']:
            if field in features:
                features[field] = label_encoders[field].fit_transform(features[field])

        # Fit and transform features based on selected model requirements
        scaler = StandardScaler()
        scaler.fit(dataset[list(required_fields[prediction_type])])
        features_scaled = scaler.transform(features.fillna(0))

        try:
            model = models[gender][prediction_type]
            probability = model.predict_proba(features_scaled)[:, 1][0]
            st.write(f'Probability of Sarcopenia: {probability:.2f}')
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

