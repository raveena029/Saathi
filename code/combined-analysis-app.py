import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from medical_extraction import initialize_processor, analyze_medical_symptoms

class CrimeAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        self.df.fillna(0, inplace=True)
        self.crime_matrix = self.df.iloc[:, 2:]
        self.crime_matrix_scaled = StandardScaler().fit_transform(self.crime_matrix)
        
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

def main():
    st.set_page_config(layout="wide", page_title="Women Safety Analysis System")
    
    st.sidebar.title("Analysis Options")
    analysis_type = st.sidebar.radio("Select Analysis Type", ["Women Crime Report", "Medical Disorder Extraction"])
    
    if analysis_type == "Women Crime Report":
        st.title("Women Crime Analysis Dashboard")
        
        try:
            crime_analysis = CrimeAnalysis('Saathi\\CrimesOnWomenData.csv')
            crime_analysis.kmeans_clustering()
            
            # Visualization Section
            st.subheader("Crime Pattern Clusters")
            cluster_fig = crime_analysis.visualize_clusters()
            st.pyplot(cluster_fig)
            
            # Query Section
            st.subheader("Crime Data Query")
            col1, col2 = st.columns(2)
            
            with col1:
                crime_type = st.selectbox(
                    "Select Crime Type",
                    options=[col for col in crime_analysis.df.columns if col not in ['State', 'Year', 'Cluster']]
                )
            
            with col2:
                year = st.selectbox(
                    "Select Year",
                    options=sorted(crime_analysis.df['Year'].unique()) if 'Year' in crime_analysis.df.columns else []
                )
            
            if st.button("Query Data"):
                results = crime_analysis.query_crime_data(crime_type, year)
                if not results.empty:
                    st.write(f"Top 5 states with highest {crime_type} cases in {year}:")
                    st.dataframe(results)
                    
                    # Visualization of query results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=results, x='State', y=crime_type)
                    plt.xticks(rotation=45)
                    plt.title(f"{crime_type} Cases by State ({year})")
                    st.pyplot(fig)
                else:
                    st.warning("No data found for the selected criteria")
                    
        except Exception as e:
            st.error(f"Error in crime analysis: {e}")
            
    else:  # Medical Disorder Extraction
        st.title("Medical Symptom Analysis System")
        processor = initialize_processor()
        
        try:
            symptoms_df = pd.read_csv("Updated_Symptom2Disease.csv")
            diseases_info_df = pd.read_csv("diseases_info.csv")
            
            if not hasattr(st.session_state, 'preprocessing_done'):
                st.session_state.preprocessing_done = processor.load_models()
            
            if not st.session_state.preprocessing_done:
                if st.sidebar.button("Start Preprocessing"):
                    with st.spinner("Preprocessing data..."):
                        metrics = processor.preprocess_datasets(symptoms_df, diseases_info_df)
                        st.session_state.preprocessing_done = True
                        if metrics:
                            st.sidebar.success("Preprocessing complete!")
                            st.sidebar.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
                            st.sidebar.metric("Test Cases", metrics['test_size'])
            else:
                st.sidebar.success("System Ready!")
            
            st.subheader("Describe your symptoms")
            user_input = st.text_area("Please provide a detailed description of your symptoms:")
            
            if st.button("Analyze Symptoms"):
                if not st.session_state.preprocessing_done:
                    st.warning("Please run preprocessing first!")
                elif user_input:
                    analyze_medical_symptoms(user_input, processor, symptoms_df, diseases_info_df)
                else:
                    st.warning("Please enter your symptoms first.")
                    
        except Exception as e:
            st.error(f"Error in medical analysis: {e}")

if __name__ == "__main__":
    main()
