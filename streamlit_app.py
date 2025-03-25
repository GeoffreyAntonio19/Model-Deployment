import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_model(filename):
    return joblib.load(filename)

def predict_with_model(model, user_input):
    prediction_proba = model.predict_proba([user_input])[0]
    prediction = np.argmax(prediction_proba)  # Get the index of the highest probability
    return prediction, prediction_proba

def main():
    st.title('Dermatology Machine Learning')
    st.write('This app uses Machine Learning to predict dermatological conditions.')

    # User input sliders
    features = [
        'Erythema', 'Scaling', 'Definite Borders', 'Itching', 'Koebner Phenomenon', 'Polygonal Papules',
        'Follicular Papules', 'Oral Mucosal Involvement', 'Knee and Elbow Involvement', 'Scalp Involvement',
        'Family History', 'Melanin Incontinence', 'Eosinophils Infiltrate', 'PNL Infiltrate',
        'Fibrosis Papillary Dermis', 'Exocytosis', 'Acanthosis', 'Hyperkeratosis', 'Parakeratosis',
        'Clubbing Rete Ridges', 'Elongation Rete Ridges', 'Thinning Suprapapillary Epidermis',
        'Spongiform Pustule', 'Munro Microabcess', 'Focal Hypergranulosis', 'Disappearance Granular Layer',
        'Vacuolisation Damage Basal Layer', 'Spongiosis', 'Saw Tooth Appearance Retes',
        'Follicular Horn Plug', 'Perifollicular Parakeratosis', 'Inflammatory Mononuclear Infiltrate',
        'Band Like Infiltrate', 'Age'
    ]
    
    user_input = [st.slider(feature, min_value=0, max_value=3, value=1) for feature in features[:-1]]
    user_input.append(st.slider('Age', min_value=0, max_value=75, value=40))
    
    model_filename = 'trained_model.pkl'
    model = load_model(model_filename)
    prediction, prediction_proba = predict_with_model(model, user_input)
    
    # Display prediction
    st.subheader('Predicted Condition')
    dermatology_classes = np.array(['Psoriasis', 'Seborrheic Dermatitis', 'Lichen Planus',
                                    'Pityriasis Rosea', 'Chronic Dermatitis', 'Pityriasis Rubra Pilaris'])
    st.success(str(dermatology_classes[prediction]))
    
    # Display prediction probabilities
    df_prediction_proba = pd.DataFrame([prediction_proba], columns=dermatology_classes)
    df_prediction_proba = df_prediction_proba.T.rename(columns={0: "Probability"}).sort_values(by="Probability", ascending=False)
    
    st.subheader('Prediction Probabilities')
    st.dataframe(df_prediction_proba.style.format('{:.4f}'))

if __name__ == "__main__":
    main()
