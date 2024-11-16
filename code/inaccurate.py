import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine
import re
from collections import Counter
from sklearn.metrics import silhouette_score

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize preprocessors
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Create a basic medical/crime vocabulary for spell checking
DOMAIN_VOCABULARY = {
    'healthcare': {
        'fever', 'cough', 'headache', 'pain', 'nausea', 'vomiting', 'diarrhea',
        'fatigue', 'weakness', 'dizziness', 'chest', 'throat', 'stomach', 'back',
        'joint', 'muscle', 'skin', 'rash', 'breathing', 'blood', 'pressure', 
        'heart', 'lungs', 'head', 'neck', 'arms', 'legs', 'chronic', 'acute',
        'severe', 'mild', 'moderate', 'symptoms', 'condition', 'disease',
        'infection', 'inflammation', 'swelling', 'temperature', 'pulse'
    },
    'crime': {
        'theft', 'robbery', 'assault', 'burglary', 'fraud', 'vandalism',
        'violence', 'weapon', 'suspect', 'victim', 'witness', 'evidence',
        'damage', 'stolen', 'breaking', 'entering', 'threat', 'criminal',
        'police', 'report', 'emergency', 'incident', 'investigation', 'scene',
        'property', 'vehicle', 'injury', 'dangerous', 'suspicious', 'activity'
    }
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

def spell_correction(text, domain):
    """Correct spelling using edit distance and domain-specific vocabulary."""
    words = text.lower().split()
    corrected_words = []
    
    domain_vocab = DOMAIN_VOCABULARY[domain]
    
    for word in words:
        if word in domain_vocab:
            corrected_words.append(word)
            continue
            
        # Find closest word in domain vocabulary
        candidates = []
        for vocab_word in domain_vocab:
            distance = calculate_edit_distance(word, vocab_word)
            if distance <= 2:  # Allow up to 2 character differences
                candidates.append((vocab_word, distance))
        
        if candidates:
            corrected_word = min(candidates, key=lambda x: x[1])[0]
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def preprocess_text(text, domain):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Apply domain-specific spell correction
    text = spell_correction(text, domain)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words, stem and lemmatize
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) 
             for token in tokens 
             if token not in stop_words]
    
    return ' '.join(tokens)

def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1.toarray().flatten(), vec2.toarray().flatten())

def find_optimal_k(X, y, max_k=10):
    distortions = []
    K = range(1, max_k + 1)
    
    for k in K:
        if k == 1:
            distortions.append(0)
            continue
            
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        scores = silhouette_score(X, model.predict(X))
        distortions.append(scores)
    
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Elbow Method for Optimal k')
    return fig, np.argmax(distortions) + 1

def plot_dendrogram(X, labels):
    linkage_matrix = linkage(X, method='ward')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Categories')
    plt.ylabel('Distance')
    plt.tight_layout()
    return fig

def analyze_data(df, user_input, domain, domain_name):
    # Preprocess the dataset
    df['processed_text'] = df[domain['symptom_col']].apply(
        lambda x: preprocess_text(x, domain_name))
    
    # Vectorize text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[domain['label_col']])
    
    # Find optimal k
    elbow_fig, optimal_k = find_optimal_k(X.toarray(), y)
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X, y)
    
    # Process user input
    processed_input = preprocess_text(user_input, domain_name)
    input_vector = vectorizer.transform([processed_input])
    
    # Get prediction and similarity scores
    prediction = knn.predict(input_vector)
    predicted_label = le.inverse_transform(prediction)[0]
    
    # Calculate similarity scores for all entries
    similarities = []
    for idx, row in df.iterrows():
        vec = vectorizer.transform([row['processed_text']])
        similarity = calculate_cosine_similarity(input_vector, vec)
        similarities.append((row[domain['label_col']], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_label, similarities, elbow_fig

def main():
    st.title("Analysis System")
    
    # Domain configurations
    domains = {
        'healthcare': {
            'file_path': r"C:\Users\ravee\Downloads\dataset.csv",
            'symptom_col': 'symptoms',
            'label_col': 'disease',
            'info_cols': ['risk level', 'doctor', 'medication']
        },
        'crime': {
            'file_path': r"C:\Users\ravee\Downloads\crime_dataset.csv",
            'symptom_col': 'description',
            'label_col': 'crime_type',
            'info_cols': ['severity', 'department', 'action_required']
        }
    }
    
    # Sidebar for domain selection
    st.sidebar.title("Settings")
    domain_choice = st.sidebar.radio("Select Domain", ['Healthcare', 'Crime'])
    
    # Set active domain
    domain_name = domain_choice.lower()
    active_domain = domains[domain_name]
    
    try:
        df = pd.read_csv(active_domain['file_path'])
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    # Main interface
    st.subheader(f"Describe the {domain_choice.lower()} situation")
    user_input = st.text_area("Please provide a detailed description:")
    
    if st.button("Analyze"):
        if user_input:
            # Show the spell-corrected input
            corrected_input = spell_correction(user_input, domain_name)
            if corrected_input != user_input:
                st.info(f"Corrected text: {corrected_input}")
            
            # Analyze the input
            predicted_label, similarities, elbow_fig = analyze_data(
                df, corrected_input, active_domain, domain_name)
            
            # Get information for the predicted case
            case_info = df[df[active_domain['label_col']] == predicted_label].iloc[0]
            
            # Display results
            st.success("Analysis Complete!")
            
            # Display primary prediction
            st.write(f"**Predicted {active_domain['label_col'].replace('_', ' ').title()}:** {predicted_label}")
            
            # Display similarity scores
            st.subheader("Similarity Analysis")
            st.write("Top 3 matching cases:")
            for label, score in similarities[:3]:
                st.write(f"- {label}: {score:.2%} match")
            
            # Display additional information
            for col in active_domain['info_cols']:
                st.write(f"**{col.replace('_', ' ').title()}:** {case_info[col]}")
            
            # Show elbow curve
            st.subheader("Model Analysis")
            st.pyplot(elbow_fig)
            
            if domain_choice.lower() == 'healthcare':
                st.warning("""
                Please note: This is just a preliminary analysis and should not be considered as a definitive diagnosis. 
                Please consult with a healthcare professional for proper medical advice and treatment.
                """)
            else:
                st.warning("""
                Please note: This is just a preliminary analysis and should not be considered as official legal advice. 
                Please consult with law enforcement or legal professionals for proper guidance.
                """)
                
        else:
            st.warning("Please enter a description first.")

if __name__ == "__main__":
    main()
