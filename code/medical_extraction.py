
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine
import seaborn as sns
import re
import joblib
import os

# Initialize preprocessors
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Medical vocabulary
MEDICAL_VOCABULARY = {
    'fever', 'cough', 'sore', 'throat', 'runny', 'stuffy', 'nose', 'muscle', 'aches', 'headache', 
    'fatigue', 'mucus', 'production', 'shortness', 'breath', 'chest', 'pain', 'nausea', 'vomiting', 
    'lightheadedness', 'sweating', 'weakness', 'numbness', 'body', 'confusion', 'difficulty', 'speaking', 
    'trouble', 'seeing', 'eye', 'severe', 'lump', 'unexplained', 'weight', 'loss', 'changes', 
    'bowel', 'bladder', 'habits', 'persistent', 'indigestion', 'bleeding', 'discharge', 'increased', 
    'thirst', 'frequent', 'urination', 'blurred', 'vision', 'cuts', 'slow', 'heal', 'memory', 
    'thinking', 'personality', 'behavior', 'stiffness', 'swelling', 'inflammation', 'joints', 
    'conjunctivitis', 'pink', 'liver', 'damage', 'death', 'dependence', 'alcohol', 'tolerance', 
    'withdrawal', 'symptoms', 'impaired', 'judgment', 'cancer', 'itchy', 'palate', 'pale', 
    'skin', 'anxiety', 'fear', 'lower', 'abdomen', 'right', 'appendix', 'social', 'interaction', 
    'repetitive', 'behaviors', 'restricted', 'interests', 'wheezing', 'tightness', 'sensitivity', 
    'light', 'mood', 'swings', 'mania', 'depression', 'blood', 'urine', 'burning', 'incontinence', 
    'redness', 'warmth', 'tenderness', 'bursa', 'sac', 'fluid', 'vaginal', 'sex', 'abdominal', 
    'bloating', 'constipation', 'upper', 'indigestion', 'diarrhea', 'sputum', 'chills', 'night', 
    'rash', 'bite', 'tingling', 'hydrophobia', 'aerophobia', 'jaundice', 'cramps', 'nerve', 
    'sweats', 'urinating', 'burn', 'twisted', 'veins', 'rough', 'growths', 'itching', 'vagina', 
    'penis', 'erection', 'pressure', 'fluid', 'increased', 'glaucoma', 'optic', 'nerve', 'heart', 
    'kidneys', 'tremors', 'coordination', 'numbness', 'tingling', 'hips', 'erectile', 'tissue', 
    'cold', 'intolerance', 'hoarseness', 'density', 'bulge', 'aorta', 'autoimmune', 'bronchitis', 
    'emphysema', 'cholesterol', 'halos', 'redness', 'joint', 'posture', 'speech', 'menstrual', 
    'cramps', 'fertility', 'acne', 'midsection', 'cysts', 'darkening', 'spots', 'gastrointestinal', 
    'problems', 'insomnia', 'metabolism', 'dryness', 'vaginal', 'bleeding', 'appetite', 'swelling', 
    'back', 'bloating', 'pelvic', 'urinary', 'frequency', 'fullness', 'menstrual', 'fibroid', 
    'uterus', 'armpit', 'swelling', 'breast', 'discharge', 'fetal', 'swelling', 'hands', 'contractions', 
    'movement', 'hydration', 'rash', 'butterfly', 'itching', 'cracking', 'pitting', 'dry', 'facial', 
    'lesions', 'sun', 'exposure', 'hair', 'digestive', 'cold', 'neck', 'mobility'
}

def calculate_edit_distance(word1, word2):
    """Calculate minimum edit distance between two words."""
    if len(word1) < len(word2):
        return calculate_edit_distance(word2, word1)

    if len(word2) == 0:
        return len(word1)

    previous_row = range(len(word2) + 1)
    for i, c1 in enumerate(word1):
        current_row = [i + 1]
        for j, c2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def spell_correction(text):
    """Correct spelling using edit distance and medical vocabulary."""
    words = text.lower().split()
    corrected_words = []
    
    for word in words:
        if word in MEDICAL_VOCABULARY:
            corrected_words.append(word)
            continue
            
        candidates = []
        for vocab_word in MEDICAL_VOCABULARY:
            distance = calculate_edit_distance(word, vocab_word)
            if distance <= 2:  # Allow up to 2 character differences
                candidates.append((vocab_word, distance))
        
        if candidates:
            corrected_word = min(candidates, key=lambda x: x[1])[0]
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def preprocess_text(text):
    """Preprocess text with medical context-specific spell correction."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Apply medical spell correction
    text = spell_correction(text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words, stem and lemmatize
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) 
             for token in tokens 
             if token not in stop_words]
    
    return ' '.join(tokens)

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return 1 - cosine(vec1.toarray().flatten(), vec2.toarray().flatten())

def find_optimal_k(X, y, max_k=10):
    """Find optimal k using elbow method with silhouette score."""
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    for k in K:
        clusterer = AgglomerativeClustering(n_clusters=k)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, silhouette_scores, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Find optimal k (highest silhouette score)
    optimal_k = K[np.argmax(silhouette_scores)]
    
    return fig, optimal_k
class DataProcessor:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.knn_model = None
        self.optimal_k = None
        self.preprocessed = False
        self.processed_symptoms = None
        self.processed_diseases = None
    
    def load_models(self):
        """Load pre-trained models if they exist."""
        try:
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.knn_model = joblib.load('models/knn_model.pkl')
            self.preprocessed = True
            return True
        except:
            return False
    
    
    def preprocess_datasets(self, symptoms_df, diseases_info_df):
        """Preprocess datasets and train models."""
        if not self.preprocessed:
            try:
                # Create processed_text column first
                symptoms_df['processed_text'] = symptoms_df['symptoms'].apply(preprocess_text)
                
                # Vectorize text
                self.vectorizer = TfidfVectorizer()
                X = self.vectorizer.fit_transform(symptoms_df['processed_text'])
                
                # Encode labels
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(symptoms_df['disease'])
                
                # Find optimal k
                _, self.optimal_k = find_optimal_k(X.toarray(), y)
                
                # Train KNN
                self.knn_model = KNeighborsClassifier(n_neighbors=self.optimal_k, weights='uniform')

                self.knn_model.fit(X, y)
                
                # Save models
                if not os.path.exists('models'):
                    os.makedirs('models')
                joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
                joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
                joblib.dump(self.knn_model, 'models/knn_model.pkl')
                
                # Store processed data
                self.processed_symptoms = symptoms_df
                self.processed_diseases = diseases_info_df
                self.preprocessed = True
                
                # Calculate and store evaluation metrics
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                y_pred = self.knn_model.predict(X_test)
                
                return {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'test_size': len(y_test)
                }
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                return None
        return None

@st.cache_resource
def initialize_processor():
    """Initialize and return a DataProcessor instance with caching."""
    return DataProcessor()
def analyze_medical_symptoms(user_input, processor, symptoms_df, diseases_info_df):
    """Analyze user symptoms and display results with detailed metrics."""
    with st.spinner("Analyzing symptoms..."):
        # Process input and get prediction
        processed_input = preprocess_text(user_input)
        input_vector = processor.vectorizer.transform([processed_input])
        prediction = processor.knn_model.predict(input_vector)
        predicted_disease = processor.label_encoder.inverse_transform(prediction)[0]
        
        # Get disease information
        disease_info = diseases_info_df[
            diseases_info_df['disease'] == predicted_disease]
        if disease_info.empty:
            st.error(f"No information found for disease: {predicted_disease}")
            return

        if len(disease_info) == 0:
            st.error(f"No information found for disease: {predicted_disease}")
            return
            
        disease_info = disease_info.iloc[0]
        
        # Use processed_symptoms from the processor
        disease_symptoms = processor.processed_symptoms[
            processor.processed_symptoms['disease'] == predicted_disease]['symptoms'].iloc[0]
        
        processed_disease_symptoms = preprocess_text(disease_symptoms)
        disease_vector = processor.vectorizer.transform([processed_disease_symptoms])
        symptom_match_score = calculate_cosine_similarity(input_vector, disease_vector)
        
        # Calculate model metrics using processed data
        y_true = (processor.label_encoder.transform(processor.processed_symptoms['disease']) == 
                 processor.label_encoder.transform([predicted_disease])[0]).astype(int)
        y_pred_proba = processor.knn_model.predict_proba(
            processor.vectorizer.transform(processor.processed_symptoms['processed_text']))
        y_pred = (y_pred_proba[:, processor.label_encoder.transform([predicted_disease])[0]] >= 0.5).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"Predicted Condition: {predicted_disease}")
            st.info("Disease Information")
            st.write(f"**Medication:** {disease_info['medication']}")
            st.write(f"**Doctor:** {disease_info['doctor']}")
            st.write(f"**Risk Level:** {disease_info['risk_level']}")
        
        with col2:
            st.info("Analysis Metrics")
            st.write(f"**Symptom Match Score:** {symptom_match_score:.2%}")
            st.write(f"**Model Accuracy:** {accuracy:.2%}")
            st.write(f"**Precision:** {precision:.2%}")
            st.write(f"**Recall:** {recall:.2%}")
            st.write(f"**F1 Score:** {f1:.2%}")
            
def main():
    st.set_page_config(layout="wide")
    
    st.title("Medical Symptom Analysis System")
    
    # Initialize processor with caching
    processor = initialize_processor()
    
    # Session state for tracking preprocessing status
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = processor.load_models()
    
    try:
        # Load datasets with correct file paths
        symptoms_df = pd.read_csv("C:\\Users\\ravee\\Downloads\\Updated_Symptom2Disease.csv")
        diseases_info_df = pd.read_csv("C:\\Users\\ravee\\Downloads\\diseases_info.csv")
        
        # Verify column names
        required_symptom_cols = ['disease', 'symptoms']
        required_disease_cols = ['disease', 'medication', 'doctor', 'risk_level']
        
        if not all(col in symptoms_df.columns for col in required_symptom_cols):
            st.error("Symptoms dataset missing required columns: disease, symptoms")
            return
            
        if not all(col in diseases_info_df.columns for col in required_disease_cols):
            st.error("Diseases info dataset missing required columns: disease, medication, doctor, risk_level")
            return
        
        # Rest of the main function remains the same
        st.sidebar.title("Analysis Options")
        
        if not st.session_state.preprocessing_done:
            if st.sidebar.button("Start Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    metrics = processor.preprocess_datasets(
                        symptoms_df, diseases_info_df)
                    st.session_state.preprocessing_done = True
                    if metrics:
                        st.sidebar.success("Preprocessing complete!")
                        st.sidebar.metric(
                            "Model Accuracy", 
                            f"{metrics['accuracy']:.2%}")
                        st.sidebar.metric(
                            "Test Cases", 
                            metrics['test_size'])
        else:
            st.sidebar.success("System Ready!")
    
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return
    
    # Input area
    st.subheader("Describe your symptoms")
    user_input = st.text_area(
        "Please provide a detailed description of your symptoms:")
    
    if st.button("Analyze Symptoms"):
        if not st.session_state.preprocessing_done:
            st.warning("Please run preprocessing first!")
            return
            
        if user_input:
            analyze_medical_symptoms(
                user_input, processor, symptoms_df, diseases_info_df)
        else:
            st.warning("Please enter your symptoms first.")

if __name__ == "__main__":
    main()
