import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
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
from jellyfish import soundex
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize preprocessors
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Medical vocabulary (keeping your existing MEDICAL_VOCABULARY)
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
def get_soundex(text):
    """Calculate Soundex codes for words in text."""
    words = text.lower().split()
    return [soundex(word) for word in words if word.isalpha()]

def calculate_phonetic_similarity(text1, text2):
    """Calculate similarity based on Soundex codes."""
    soundex1 = get_soundex(text1)
    soundex2 = get_soundex(text2)
    
    # Convert lists to Counter objects for comparison
    counter1 = Counter(soundex1)
    counter2 = Counter(soundex2)
    
    # Calculate intersection and union
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())
    
    # Calculate Jaccard similarity
    return intersection / union if union > 0 else 0

def calculate_symptom_match_percentage(user_symptoms, disease_symptoms):
    """Calculate the percentage of matching symptoms."""
    user_symptoms_set = set(user_symptoms.lower().split())
    disease_symptoms_set = set(disease_symptoms.lower().split())
    
    # Calculate intersection and percentage
    matching_symptoms = user_symptoms_set.intersection(disease_symptoms_set)
    total_disease_symptoms = len(disease_symptoms_set)
    
    return (len(matching_symptoms) / total_disease_symptoms * 100) if total_disease_symptoms > 0 else 0

def calculate_detailed_metrics(y_true, y_pred, y_pred_proba):
    """Calculate detailed classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


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
    if vec1.shape[1] != vec2.shape[1]:
        raise ValueError("Vectors must have the same dimensions")
    return 1 - cosine(vec1.toarray().flatten(), vec2.toarray().flatten())

def find_optimal_k(X, y, max_k=10):
    """Find optimal k using elbow method with silhouette score."""
    silhouette_scores = []
    K = range(2, min(max_k + 1, len(np.unique(y))))
    
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
            if not os.path.exists('models'):
                return False
            
            model_files = ['vectorizer.pkl', 'label_encoder.pkl', 'knn_model.pkl', 'processed_symptoms.pkl', 'processed_diseases.pkl']
            if not all(os.path.exists(f'models/{file}') for file in model_files):
                return False
                
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.knn_model = joblib.load('models/knn_model.pkl')
            self.processed_symptoms = joblib.load('models/processed_symptoms.pkl')
            self.processed_diseases = joblib.load('models/processed_diseases.pkl')
            self.preprocessed = True
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def preprocess_datasets(self, symptoms_df, diseases_info_df):
        """Preprocess datasets and train models."""
        if not self.preprocessed:
            try:
                # Store the dataframes
                self.processed_symptoms = symptoms_df.copy()
                self.processed_diseases = diseases_info_df.copy()
                
                # Create processed_text column
                self.processed_symptoms['processed_text'] = self.processed_symptoms['symptoms'].apply(preprocess_text)
                
                # Vectorize text
                self.vectorizer = TfidfVectorizer()
                X = self.vectorizer.fit_transform(self.processed_symptoms['processed_text'])
                
                # Encode labels
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(self.processed_symptoms['disease'])
                
                # Find optimal k
                _, self.optimal_k = find_optimal_k(X.toarray(), y)
                
                # Train KNN
                self.knn_model = KNeighborsClassifier(n_neighbors=self.optimal_k, weights='uniform')
                self.knn_model.fit(X, y)
                
                # Save models and processed data
                if not os.path.exists('models'):
                    os.makedirs('models')
                joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
                joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
                joblib.dump(self.knn_model, 'models/knn_model.pkl')
                joblib.dump(self.processed_symptoms, 'models/processed_symptoms.pkl')
                joblib.dump(self.processed_diseases, 'models/processed_diseases.pkl')
                
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

def analyze_medical_symptoms(user_input, processor, symptoms_df, diseases_info_df):
    """Analyze user symptoms with enhanced metrics and phonetic matching."""
    try:
        with st.spinner("Analyzing symptoms..."):
            if not processor.preprocessed:
                st.error("Models not properly initialized. Please run preprocessing first.")
                return
                
            # Process input and get prediction
            processed_input = preprocess_text(user_input)
            input_vector = processor.vectorizer.transform([processed_input])
            
            # Get prediction and probabilities
            prediction = processor.knn_model.predict(input_vector)
            probabilities = processor.knn_model.predict_proba(input_vector)[0]
            predicted_disease = processor.label_encoder.inverse_transform(prediction)[0]
            
            # Get disease information
            disease_info = diseases_info_df[diseases_info_df['disease'] == predicted_disease]
            if len(disease_info) == 0:
                st.error(f"No information found for disease: {predicted_disease}")
                return
                
            disease_info = disease_info.iloc[0]
            
            # Get ALL matching symptoms for the disease
            matching_symptoms = symptoms_df[symptoms_df['disease'] == predicted_disease]
            if len(matching_symptoms) == 0:
                st.error(f"No symptoms found for disease: {predicted_disease}")
                return
            
            # Aggregate all symptoms while preserving bi-words
            all_symptoms = []
            for symptom_text in matching_symptoms['symptoms']:
                # Split by comma or and, but preserve important bi-words
                symptom_list = re.split(r',\s*|\s+and\s+', symptom_text)
                all_symptoms.extend(symptom_list)
            
            # Remove duplicates while preserving order
            unique_symptoms = list(dict.fromkeys(all_symptoms))
            combined_symptoms = " ; ".join(unique_symptoms)
            
            # Calculate cosine similarity using TF-IDF vectors
            disease_vectors = processor.vectorizer.transform(matching_symptoms['symptoms'])
            similarities = [calculate_cosine_similarity(input_vector, disease_vector) 
                          for disease_vector in disease_vectors]
            max_tfidf_similarity = max(similarities)
            
            # Calculate phonetic similarity with all symptom sets
            phonetic_similarities = [calculate_phonetic_similarity(user_input, symptoms) 
                                   for symptoms in matching_symptoms['symptoms']]
            max_phonetic_similarity = max(phonetic_similarities)
            
            # Calculate symptom match percentage considering all symptom sets
            symptom_match_percentages = [calculate_symptom_match_percentage(user_input, symptoms) 
                                       for symptoms in matching_symptoms['symptoms']]
            max_symptom_match = max(symptom_match_percentages)
            
            # Get confidence score
            predicted_class_index = processor.label_encoder.transform([predicted_disease])[0]
            confidence_score = probabilities[predicted_class_index]
            
            # Display results in an organized layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Predicted Condition: {predicted_disease}")
                st.info("Disease Information")
                st.write(f"**Medication:** {disease_info['medication']}")
                st.write(f"**Doctor:** {disease_info['doctor']}")
                st.write(f"**Risk Level:** {disease_info['risk_level']}")
                
                # Add symptoms comparison
                st.info("Symptoms Comparison")
                st.write("**Your symptoms:**")
                st.write(user_input)
                st.write("**All typical symptoms:**")
                # Display each symptom on a new line for better readability
                for symptom in unique_symptoms:
                    st.write(f"• {symptom.strip()}")
            
            with col2:
                st.info("Analysis Metrics")
                
                # Confidence and Similarity Metrics
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Confidence Score", f"{confidence_score:.1%}")
                    st.metric("TF-IDF Similarity", f"{max_tfidf_similarity:.1%}")
                    st.metric("Phonetic Similarity", f"{max_phonetic_similarity:.1%}")
                
                with metrics_col2:
                    st.metric("Symptom Match", f"{max_symptom_match:.1%}")
                    st.metric("Model Accuracy", f"{processor.knn_model.score(processor.vectorizer.transform(processor.processed_symptoms['processed_text']), processor.label_encoder.transform(processor.processed_symptoms['disease'])):.1%}")
                
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)
        return


@st.cache_resource
def initialize_processor():
    """Initialize and return a DataProcessor instance with caching."""
    return DataProcessor()

def main():
    st.set_page_config(layout="wide")
    
    st.title("Medical Symptom Analysis System")
    
    # Initialize processor with caching
    processor = initialize_processor()
    
    # Session state for tracking preprocessing status
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = processor.load_models()
    
    try:
        # Load datasets
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
        
        st.sidebar.title("Analysis Options")
        
        if not st.session_state.preprocessing_done:
            if st.sidebar.button("Start Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    metrics = processor.preprocess_datasets(symptoms_df, diseases_info_df)
                    if metrics:
                        st.session_state.preprocessing_done = True
                        st.sidebar.success("Preprocessing complete!")
                        st.sidebar.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
                        st.sidebar.metric("Test Cases", metrics['test_size'])
        else:
            st.sidebar.success("System Ready!")
    
        # Input area
        st.subheader("Describe your symptoms")
        user_input = st.text_area("Please provide a detailed description of your symptoms:")
        
        if st.button("Analyze Symptoms"):
            if not st.session_state.preprocessing_done:
                st.warning("Please run preprocessing first!")
                return
                
            if not user_input:
                st.warning("Please enter your symptoms first.")
                return
                
            analyze_medical_symptoms(user_input, processor, symptoms_df, diseases_info_df)
    
    except Exception as e:
        st.error(f"Error in main function: {e}")
        return

if __name__ == "__main__":
    main()
