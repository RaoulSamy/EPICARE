import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import io
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import joblib
from groq import Groq
from pymongo import MongoClient

# Define the Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# MongoDB client setup
client_mongo = MongoClient('mongodb://localhost:27017/')
db = client_mongo['EPICARE']  # Access the EPICARE database
collection = db['skin_cancer_queries']  # Access the skin_cancer_queries collection

# Load models
skin_non_skin_model = load_model('model/skin_non_skin_classifier.h5')
fine_tuned_classification_model = load_model('model/fine_tuned_skin_cancer_classifier.h5')
tracker_model = load_model('model/rnn_model.h5')
side_effects_model = load_model('model/skin_cancer_side_effects_model.h5')
severity_model = load_model('model/skin_cancer_severity_model.h5')

# Load the label encoders
with open('model/label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Load recommendation system
model_file_path = r"model/recommendation_system.pkl"
with open(model_file_path, 'rb') as f:
    data = joblib.load(f)

tfidf_vectorizer = data['tfidf_vectorizer']
cosine_sim = data['cosine_sim']
patient_df = data['patient_df']
hospital_df = data['hospital_df']

# Allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess image
def preprocess_image(image, target_size):
    img = Image.open(io.BytesIO(image))
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Chatbot API integration
client_groq = Groq(api_key='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')

# List of skin cancer-related keywords/topics
SKIN_CANCER_KEYWORDS = [
    "skin cancer", "melanoma", "basal cell carcinoma", "bcc", "squamous cell carcinoma",
    "scc", "dermatology", "skin lesions", "tumors", "cancer", "UV exposure", 
    "biopsy", "treatment", "radiation", "chemotherapy", "surgery", "skin diagnosis",
    "side effects", "skin cancer symptoms", "skincare"
]

# Helper function to check if the query is related to skin cancer
def is_related_to_skin_cancer(user_input):
    user_input = user_input.lower()  # Convert user input to lowercase
    for keyword in SKIN_CANCER_KEYWORDS:
        if keyword in user_input:
            return True
    return False

# Routes
@app.route('/')
def home():
    return render_template('index.html')

# Route to render the chatbot page
@app.route('/chatbot', methods=['GET'])
def chatbot_page():
    return render_template('chatbot.html')

# Chatbot Route - Checks MongoDB first, then Groq if no match found
@app.route('/chatbot', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    # Check if the query is related to skin cancer
    if not is_related_to_skin_cancer(user_input):
        return jsonify({'response': 'This chatbot is specialized in skin cancer-related topics. Please ask questions about skin cancer, its treatments, symptoms, or care.'}), 200

    # Search in MongoDB for a matching query
    query_result = collection.find_one({'query': user_input})

    if query_result:
        # If a match is found, return the response from MongoDB
        bot_response = query_result['answer']
        return jsonify({'response': bot_response})

    try:
        # If no match, use Groq API with Llama3 model for the response
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model="llama3-70b-8192"
        )

        bot_response = chat_completion.choices[0].message.content.strip()

        if not bot_response:
            return jsonify({'error': 'No valid response generated'}), 500

        # Save the new query and response to MongoDB
        collection.insert_one({'query': user_input, 'answer': bot_response})

        return jsonify({'response': bot_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Failed to generate response from Groq'}), 500

@app.route('/skincare', methods=['GET', 'POST'])
def skin_care():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            file_bytes = file.read()  # Read file in memory without saving
            
            # Preprocess for skin vs. non-skin classification (150x150)
            img_array = preprocess_image(file_bytes, target_size=(150, 150))
            skin_prediction = skin_non_skin_model.predict(img_array)
            
            if skin_prediction[0] > 0.5:  # Assuming binary classification (0=non-skin, 1=skin)
                # Preprocess for fine-tuned skin cancer classification (150x150)
                img_array = preprocess_image(file_bytes, target_size=(150, 150))
                cancer_prediction = fine_tuned_classification_model.predict(img_array)
                
                # Print shape and prediction for debugging
                print("Cancer prediction shape:", cancer_prediction.shape)
                print("Cancer prediction:", cancer_prediction)
                
                # Handle output based on shape
                if cancer_prediction.shape[1] == 2:  # Assuming 2 classes (BCC, Melanoma)
                    prob_bcc = cancer_prediction[0][0]
                    prob_melanoma = cancer_prediction[0][1]
                    
                    if prob_bcc > prob_melanoma:
                        cancer_type = 'BCC'
                    else:
                        cancer_type = 'Melanoma'
                else:
                    # Handle case for a single output node
                    prob = cancer_prediction[0][0]
                    if prob > 0.5:
                        cancer_type = 'Melanoma'
                    else:
                        cancer_type = 'BCC'
                
                return render_template('skincare.html', result=f'Skin image classified as: {cancer_type}')
            else:
                return render_template('skincare.html', result="Not a skin image. Please upload a microscopic image of skin.")
    
    return render_template('skincare.html')


# Function to get side effects and severity using the models and encoders
def get_side_effects_and_severity(cancer_type, med_type, med_name):
    # Encode user inputs
    encoded_input = np.array([
        label_encoders['Skin Cancer Type'].transform([cancer_type])[0],
        label_encoders['Medication Type'].transform([med_type])[0],
        label_encoders['Medication Name'].transform([med_name])[0]
    ]).reshape(1, -1)

    # Predict side effects
    side_effects_prediction = side_effects_model.predict(encoded_input)
    side_effects = label_encoders['Side Effects Observed'].inverse_transform([np.argmax(side_effects_prediction)])

    # Predict severity of side effects
    severity_prediction = severity_model.predict(encoded_input)
    severity = label_encoders['Severity of Side Effects'].inverse_transform([np.argmax(severity_prediction)])

    return side_effects[0], severity[0]

# Monitoring route
@app.route('/monitoring', methods=['GET', 'POST'])
def monitoring():
    if request.method == 'POST':
        # Get inputs from form
        skincare_type_input = request.form.get('skincare_type')
        medication_type_input = request.form.get('medication_type')
        medication_name_input = request.form.get('medication_name')

        try:
            # Get side effects and severity predictions
            side_effects, severity = get_side_effects_and_severity(
                skincare_type_input, medication_type_input, medication_name_input
            )
            result = f"Predicted Side Effects: {side_effects}. Predicted Severity: {severity}."
        except Exception as e:
            result = f"Error during prediction: {str(e)}"
        
        # Render result on webpage
        return render_template('monitoring.html', result=result)

    return render_template('monitoring.html')

@app.route('/tracker', methods=['GET', 'POST'])
def tracker():
    if request.method == 'POST':
        # Collect inputs from the form and convert them to numeric types or encoded categories
        age = int(request.form['age'])
        sex = 0 if request.form['sex'] == 'M' else 1  # Assuming M=0, F=1
        family_history = int(request.form['family_history'])
        
        sun_exposure_map = {'Low': 0, 'Medium': 1, 'High': 2}
        sun_exposure = sun_exposure_map[request.form['sun_exposure']]
        
        skin_type_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5}
        skin_type = skin_type_map[request.form['skin_type']]
        
        immunosuppression = int(request.form['immunosuppression'])
        cancer_type = 0 if request.form['cancer_type'] == 'BCC' else 1
        
        initial_stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        initial_stage = initial_stage_map[request.form['initial_stage']]
        
        location_map = {'Head': 0, 'Neck': 1, 'Trunk': 2, 'Arms': 3, 'Legs': 4}
        location = location_map[request.form['location']]
        
        size = float(request.form['size'])
        ulceration = int(request.form['ulceration'])
        ptch1_mutation = int(request.form['ptch1_mutation'])
        
        treatment_map = {'Surgery': 0, 'Chemotherapy': 1, 'Radiation': 2}  # Adjust as needed
        treatment = treatment_map[request.form['treatment']]

        # Assuming the model needs 24 features, pad remaining fields with 0 if necessary
        # This assumes you have 13 inputs, and we need 11 more zeros to make 24 features
        input_data = np.array([[age, sex, family_history, sun_exposure, skin_type,
                                immunosuppression, cancer_type, initial_stage,
                                location, size, ulceration, ptch1_mutation, treatment]])

        # Add zero-padding for missing features to match expected 24 input features
        padded_input = np.pad(input_data, ((0, 0), (0, 11)), 'constant')  # Pad with 11 zeros to get 24 features

        # Reshape to (batch_size, timesteps, features) -> (1, 1, 24)
        padded_input = padded_input.reshape((1, 1, 24))

        # Predict using the tracker model
        tracker_result = tracker_model.predict(padded_input)

        # Assuming the model outputs a probability for progression
        progression_probability = tracker_result[0][0]
        predicted_progression = 'Yes' if progression_probability > 0.5 else 'No'

        result = {
            'predicted_progression': predicted_progression,
            'progression_probability': f"{progression_probability:.2f}"
        }

        return render_template('tracker.html', result=result)

    return render_template('tracker.html')


def recommend_hospitals(patient_index, top_n=5):
    """Recommend hospitals for a given patient based on cosine similarity."""
    sim_scores = list(enumerate(cosine_sim[patient_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    hospital_indices = [i[0] for i in sim_scores]
    return hospital_df.iloc[hospital_indices]

def get_patient_condition(patient_id):
    """Retrieve the diagnosis for a given patient ID."""
    try:
        return patient_df.loc[patient_df['PatientID'] == int(patient_id), 'Diagnosis'].values[0]
    except IndexError:
        return "Condition not found"


@app.route('/hub', methods=['GET', 'POST'])
def hub():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        
        try:
            # Get patient index and condition
            patient_index = patient_df.index[patient_df['PatientID'] == int(patient_id)].tolist()[0]
            condition = get_patient_condition(patient_id)
            
            # Get hospital recommendations
            recommendations = recommend_hospitals(patient_index, top_n=5)
            
            # Prepare the result
            result = {
                'patient_condition': condition,
                'recommendations': recommendations.to_dict(orient='records')  # Convert DataFrame to list of dicts
            }
            
        except Exception as e:
            flash(f"Error: {str(e)}")
            return redirect(request.url)
        
        return render_template('hub.html', result=result)
    
    return render_template('hub.html')


# Main entry
if __name__ == '__main__':
    app.run(debug=True)
