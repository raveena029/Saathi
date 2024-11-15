import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams
import nltk
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, silhouette_score
from collections import Counter
from kneed import KneeLocator


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text):
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        # Lemmatization
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Stemming
        stemmed = [self.stemmer.stem(token) for token in lemmatized]
        
        # Soundex
        soundex = [jellyfish.soundex(token) for token in stemmed]
        
        # Create bigrams
        bigrams = list(ngrams(stemmed, 2))
        bigrams = [f"{bg[0]}_{bg[1]}" for bg in bigrams]
        
        # Combine all features
        all_features = stemmed + soundex + bigrams
        
        return ' '.join(all_features)

class KValueOptimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.k_range = range(1, 21)  # Test k values from 1 to 20
        
    def calculate_metrics(self):
        inertias = []
        accuracies = []
        silhouette_scores = []
        
        for k in self.k_range:
            # KMeans for inertia
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X)
            inertias.append(kmeans.inertia_)
            
            # KNN for accuracy
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X, self.y)
            accuracies.append(knn.score(self.X, self.y))
            
            # Silhouette score
            if k > 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores.append(silhouette_score(self.X, kmeans.labels_))
            else:
                silhouette_scores.append(0)
                
        return inertias, accuracies, silhouette_scores
    
    def find_optimal_k(self):
        inertias, accuracies, silhouette_scores = self.calculate_metrics()
        
        # Find elbow point using KneeLocator
        kn = KneeLocator(list(self.k_range), inertias, curve='convex', direction='decreasing')
        elbow_k = kn.knee
        
        # Find k with maximum silhouette score
        silhouette_k = self.k_range[np.argmax(silhouette_scores)]
        
        # Find k with best accuracy (considering diminishing returns)
        accuracy_differences = np.diff(accuracies)
        accuracy_k = np.where(accuracy_differences < 0.01)[0][0] + 1 if len(np.where(accuracy_differences < 0.01)[0]) > 0 else len(self.k_range)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(self.k_range, inertias, 'bx-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        if elbow_k:
            ax1.axvline(x=elbow_k, color='r', linestyle='--')
        
        # Accuracy curve
        ax2.plot(self.k_range, accuracies, 'gx-')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Method')
        ax2.axvline(x=accuracy_k, color='r', linestyle='--')
        
        # Silhouette curve
        ax3.plot(self.k_range, silhouette_scores, 'rx-')
        ax3.set_xlabel('k')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Silhouette Method')
        ax3.axvline(x=silhouette_k, color='r', linestyle='--')
        
        plt.tight_layout()
        
        return {
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'accuracy_k': accuracy_k,
            'visualization': fig
        }

def main():
    st.title("Medical Symptom Disease Classifier")
    
    # Load and preprocess data
    df = pd.read_csv("C:\\Users\\ravee\\Downloads\\Symptom2Disease.csv")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess all texts
    df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # K-value optimization
    if st.sidebar.checkbox("Show K-value Analysis"):
        st.subheader("K-value Optimization Analysis")
        
        k_optimizer = KValueOptimizer(X.toarray(), y)
        optimal_k_results = k_optimizer.find_optimal_k()
        
        st.write("Optimal k-values found:")
        st.write(f"- Elbow Method: {optimal_k_results['elbow_k']}")
        st.write(f"- Silhouette Method: {optimal_k_results['silhouette_k']}")
        st.write(f"- Accuracy Method: {optimal_k_results['accuracy_k']}")
        
        st.pyplot(optimal_k_results['visualization'])
        
        # Use the elbow method k-value as default
        k = optimal_k_results['elbow_k'] if optimal_k_results['elbow_k'] else 5
        st.write(f"Using k = {k} for classification")
    else:
        k = 5  # Default value if analysis is not shown
    
    # Perform hierarchical clustering
    if st.sidebar.checkbox("Show Dendrogram"):
        st.subheader("Hierarchical Clustering Dendrogram")
        linkage_matrix = linkage(X.toarray()[:20], method='ward')
        fig, ax = plt.subplots(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        st.pyplot(fig)
    
    # Train KNN classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    
    # Show model accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")
    
    # User input for symptom classification
    st.subheader("Enter Your Symptoms")
    user_input = st.text_area("Describe your symptoms:", height=100)
    
    if st.button("Classify Disease"):
        if user_input:
            # Preprocess user input
            processed_input = preprocessor.preprocess(user_input)
            
            # Transform using the same vectorizer
            input_vector = vectorizer.transform([processed_input])
            
            # Predict disease
            prediction = clf.predict(input_vector)
            predicted_disease = le.inverse_transform(prediction)[0]
            
            # Get confidence scores
            distances, indices = clf.kneighbors(input_vector)
            confidence = 1 / (1 + distances.mean())
            
            st.success(f"Predicted Disease: {predicted_disease}")
            st.info(f"Confidence Score: {confidence:.2f}")
            
            # Show similar cases
            st.subheader("Similar Cases:")
            similar_cases = df.iloc[indices[0]]
            for idx, case in similar_cases.iterrows():
                st.write(f"- {case['text']}")
        else:
            st.warning("Please enter your symptoms.")

if __name__ == "__main__":
    main()
