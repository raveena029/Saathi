import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import jellyfish

# NLTK stopwords download
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize and load the crime analysis data
class CrimeAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        # Convert Year column to numeric if it's not already
        self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
        
        # Define crime types mapping
        self.crime_mapping = {
            'K&A': 'Kidnapping and Abduction',
            'DD': 'Dowry Deaths',
            'AoW': 'Assault on Women',
            'AoM': 'Assault on Minors',
            'DV': 'Domestic Violence',
            'Rape': 'Rape'
        }
        
        # Prepare matrix for clustering
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.crime_matrix = self.df[numeric_columns]
        self.crime_matrix_scaled = StandardScaler().fit_transform(self.crime_matrix)
        
    def filter_crime_data(self, selected_year=None, selected_state=None, selected_crime=None):
        """Filter crime data based on user selections"""
        filtered_df = self.df.copy()
        
        if selected_year:
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
        
        if selected_state:
            filtered_df = filtered_df[filtered_df['State'] == selected_state]
        
        if selected_crime:
            full_crime_name = self.crime_mapping.get(selected_crime, selected_crime)
            filtered_df = filtered_df[filtered_df['Crime Type'] == full_crime_name]
        
        return filtered_df
    
    def rank_states_by_crime(self, selected_year=None, selected_crime=None):
        """Rank states based on crime frequency"""
        filtered_df = self.df.copy()
        
        if selected_year:
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
        
        if selected_crime:
            full_crime_name = self.crime_mapping.get(selected_crime, selected_crime)
            filtered_df = filtered_df[filtered_df['Crime Type'] == full_crime_name]
        
        # Group by state and calculate total crime count
        state_crime_ranking = filtered_df.groupby('State')['Crime Count'].sum().sort_values(ascending=False)
        
        return state_crime_ranking
    
    def compute_crime_metrics(self, df):
        """Compute accuracy, precision, recall, and F1 score"""
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data for classification
        le = LabelEncoder()
        X = df[['Year', 'Crime Count']]
        y = le.fit_transform(df['State'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use KNN for classification
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Compute metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
        
    def kmeans_clustering(self, num_clusters=4, max_iters=100, tolerance=1e-4):
        np.random.seed(42)
        initial_indices = np.random.choice(len(self.crime_matrix_scaled), num_clusters, replace=False)
        centroids = self.crime_matrix_scaled[initial_indices]
        
        for iteration in range(max_iters):
            distances = np.linalg.norm(self.crime_matrix_scaled[:, np.newaxis] - centroids, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)
            
            new_centroids = np.array([
                self.crime_matrix_scaled[cluster_assignments == k].mean(axis=0) for k in range(num_clusters)
            ])
            
            if np.all(np.abs(new_centroids - centroids) < tolerance):
                break
            centroids = new_centroids
            
        self.df['Cluster'] = cluster_assignments
        self.centroids = centroids
        
    def visualize_clusters(self):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.crime_matrix_scaled)
        
        viz_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
        viz_df['Cluster'] = self.df['Cluster']
        viz_df['State'] = self.df['State']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        for cluster_id in sorted(viz_df['Cluster'].unique()):
            cluster_data = viz_df[viz_df['Cluster'] == cluster_id]
            ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f"Cluster {cluster_id}", s=50)
            
        annotated_states = viz_df.groupby('Cluster').apply(lambda x: x.nlargest(3, 'PC1')).reset_index(drop=True)
        for _, row in annotated_states.iterrows():
            ax.annotate(
                row['State'],
                (row['PC1'], row['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
            
        plt.title('Crime Pattern Clusters Across Indian States', fontsize=14)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.legend(title='Cluster', loc='best', fontsize=10)
        plt.grid(True)
        return fig
        
    def query_crime_data(self, crime_type, year):
        if 'Year' in self.df.columns:
            filtered_data = self.df[self.df['Year'] == year]
            if crime_type in self.df.columns:
                sorted_data = filtered_data.sort_values(by=crime_type, ascending=False)
                return sorted_data[['State', crime_type]].head(5)
        return pd.DataFrame()

# Preprocessing for symptom text (used for disease classification)
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Correct spelling based on Soundex and Levenshtein distance
def correct_spelling(user_symptom, vocabulary):
    corrected_symptom = []
    for word in user_symptom.split():
        soundex_word = jellyfish.soundex(word)
        closest_word = min(vocabulary, key=lambda x: jellyfish.levenshtein_distance(word, x))
        if jellyfish.soundex(closest_word) == soundex_word:
            corrected_symptom.append(closest_word)
        else:
            corrected_symptom.append(word)
    return ' '.join(corrected_symptom)

# Function to find the optimal value of k for KNN using the Elbow method
def find_optimal_k(X_train, y_train):
    error_rates = []
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        error = np.mean(y_pred != y_train)
        error_rates.append(error)
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, 21), error_rates, marker='o', linestyle='--')
    plt.title('Elbow Method to Find Optimal k')
    plt.xlabel('k')
    plt.ylabel('Error Rate')
    st.pyplot(plt)
    optimal_k = error_rates.index(min(error_rates)) + 1
    return optimal_k

# Streamlit interface
def main():
    st.set_page_config(layout="wide", page_title="Health and Crime Analysis System")
    
    # Constant title for the page
    st.title("Saathi")
    
    st.sidebar.title("Analysis Options")
    analysis_type = st.sidebar.radio("Select Analysis Type", ["Women Crime Report", "Disease Predictor"])
    
    # Women Crime Report Section
    if analysis_type == "Women Crime Report":
        st.title("Women Crime Analysis Dashboard")
    
        try:
            crime_analysis = CrimeAnalysis(r"C:\Users\ravee\Downloads\CrimesOnWomenData.csv")
            crime_analysis.kmeans_clustering()
            
            # Visualization Section
            st.subheader("Crime Pattern Clusters")
            cluster_fig = crime_analysis.visualize_clusters()
            st.pyplot(cluster_fig)
            
            # Query Section
            st.subheader("Crime Data Query")

            # Text input for the query
            STATES = [
                'ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHHATTISGARH', 
                'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 
                'JHARKHAND', 'KARNATAKA', 'KERALA', 'MADHYA PRADESH', 'MAHARASHTRA', 
                'MANIPUR', 'MEGHALAYA', 'MIZORAM', 'NAGALAND', 'ODISHA', 'PUNJAB', 
                'RAJASTHAN', 'SIKKIM', 'TAMIL NADU', 'TRIPURA', 'UTTAR PRADESH', 
                'UTTARAKHAND', 'WEST BENGAL', 'A & N ISLANDS', 'CHANDIGARH', 
                'D & N HAVELI', 'DAMAN & DIU', 'LAKSHADWEEP', 'PUDUCHERRY'
            ]
            
            YEARS = list(range(2001, 2012))
            
            CRIME_TYPES = {
                'K&A': 'Kidnapping and Abduction',
                'DD': 'Dowry Deaths',
                'AoW': 'Assault on Women',
                'AoM': 'Assault on Minors',
                'DV': 'Domestic Violence',
                'Rape': 'Rape'
            }
            
            try:
                # Initialize crime analysis
                crime_analysis = CrimeAnalysis(r"C:\Users\ravee\Downloads\CrimesOnWomenData.csv")
                
                # Create dropdowns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_year = st.selectbox("Select Year", [None] + YEARS)
                
                with col2:
                    selected_state = st.selectbox("Select State", [None] + STATES)
                
                with col3:
                    selected_crime = st.selectbox("Select Crime Type", [None] + list(CRIME_TYPES.keys()))
                
                # Filter and display data
                if st.button("Show Data"):
                    # If only year is selected, show all states and crimes for that year
                    if selected_year and not selected_state and not selected_crime:
                        filtered_data = crime_analysis.filter_crime_data(selected_year=selected_year)
                        st.dataframe(filtered_data)
                    else:
                        filtered_data = crime_analysis.filter_crime_data(
                            selected_year=selected_year, 
                            selected_state=selected_state, 
                            selected_crime=selected_crime
                        )
                        st.dataframe(filtered_data)
                
                # Ranking Section
                st.subheader("State Crime Ranking")
                if st.button("Rank States"):
                    state_ranking = crime_analysis.rank_states_by_crime(
                        selected_year=selected_year, 
                        selected_crime=selected_crime
                    )
                    st.dataframe(state_ranking)
                
                # Metrics Computation
                if st.button("Compute Crime Classification Metrics"):
                    metrics = crime_analysis.compute_crime_metrics(crime_analysis.df)
                    st.write("Crime Classification Metrics:")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}")
            except Exception as e:
                st.error(f"Error in crime analysis: {e}")
        except Exception as e:
            st.error(f"Error loading crime analysis data: {e}")
            
    # Disease Classification Section
    else:  # Disease Classification
        st.title("Disease Classification Based on Symptoms")
        dataset_path = r"C:\Users\ravee\Downloads\dataset.csv"
        st.write("Loading the dataset...")
        df = pd.read_csv(dataset_path)
        
        # Preprocess the dataset
        df['symptoms'] = df['symptoms'].apply(preprocess_text)
        le = LabelEncoder()
        df['disease'] = le.fit_transform(df['disease'])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['symptoms']).toarray()
        y = df['disease']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("Finding the optimal value of k using the Elbow Method...")
        optimal_k = find_optimal_k(X_train, y_train)
        st.write(f"The optimal value of k is: {optimal_k}")

        # KNN Model Training
        knn = KNeighborsClassifier(n_neighbors=optimal_k)
        knn.fit(X_train, y_train)
        
        # User Input for Symptoms
        user_symptom = st.text_input("Enter your symptoms:")

        # Button to start the search process
        if st.button("Search"):
            if user_symptom:
                # Preprocess and correct spelling
                user_symptom_preprocessed = preprocess_text(user_symptom)
                vocabulary = vectorizer.get_feature_names_out()
                corrected_symptom = correct_spelling(user_symptom_preprocessed, vocabulary)

                # Display suggestion if the symptom was corrected
                if corrected_symptom != user_symptom_preprocessed:
                    st.write(f"Did you mean: {corrected_symptom}?")

                # Vectorize the user input
                user_vector = vectorizer.transform([corrected_symptom]).toarray()

                # Classification
                prediction = knn.predict(user_vector)
                predicted_disease = le.inverse_transform(prediction)[0]

                # Retrieve details from the dataset
                disease_data = df[df['disease'] == prediction[0]].iloc[0]
                medication = disease_data['medication']
                doctor = disease_data['doctor']
                risk_level = disease_data['risk_level']
                symptoms = disease_data['symptoms']

                # Similarity Score
                similarity_score = 1 - cdist(user_vector, [X_train[y_train == prediction[0]].mean(axis=0)], metric='cosine')[0][0]

                # Display Results
                st.write(f"**Predicted Disease**: {predicted_disease}")
                st.write(f"**Medication**: {medication}")
                st.write(f"**Doctor**: {doctor}")
                st.write(f"**Risk Level**: {risk_level}")
                st.write(f"**Symptoms of the Predicted Disease**: {symptoms}")
                st.write(f"**Similarity Score**: {similarity_score:.2f}")

                # Model Evaluation
                y_pred_test = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_test)
                f1 = f1_score(y_test, y_pred_test, average='weighted')

                st.write(f"**Model Accuracy**: {accuracy:.2f}")
                st.write(f"**Model F1 Score**: {f1:.2f}")
            else:
                st.warning("Please enter symptoms.")

if __name__ == '__main__':
    main()
