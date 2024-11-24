import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    """
    Comprehensive data preprocessing
    """
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Convert state names to lowercase and strip whitespace
    processed_df['State'] = processed_df['State'].str.lower().str.strip()
    
    # Remove special characters from state names
    processed_df['State'] = processed_df['State'].str.replace('&', 'and')
    
    # Handle missing values
    numeric_columns = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
    processed_df[numeric_columns] = processed_df[numeric_columns].fillna(0)
    
    # Add population-adjusted crime rates (if population data available)
    # For now, we'll work with absolute numbers
    
    # Remove outliers using IQR method
    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df
    
    processed_df = remove_outliers(processed_df, numeric_columns)
    
    # Log transform to handle skewed distributions
    for col in numeric_columns:
        processed_df[col] = np.log1p(processed_df[col])
    
    return processed_df

def prepare_data(df):
    """
    Prepare the data for clustering by calculating average crime rates per state
    """
    # Preprocess the data first
    processed_df = preprocess_data(df)
    
    # List of crime columns
    crime_columns = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
    
    # Calculate mean crime rates for each state
    state_crimes = processed_df.groupby('State')[crime_columns].mean().reset_index()
    
    # Add total crime rate for additional context
    state_crimes['Total_Crime_Rate'] = state_crimes[crime_columns].sum(axis=1)
    
    return state_crimes

def find_optimal_clusters(data, max_clusters=10):
    """
    Find optimal number of clusters using elbow method and silhouette score
    """
    X = data.drop(['State', 'Total_Crime_Rate'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Return the number of clusters with highest silhouette score
    return silhouette_scores.index(max(silhouette_scores)) + 2

def perform_clustering(data, n_clusters):
    """
    Perform K-means clustering and return labeled data
    """
    X = data.drop(['State', 'Total_Crime_Rate'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the original data
    result = data.copy()
    result['Cluster'] = clusters
    
    # Sort states by total crime rate within each cluster
    result = result.sort_values(['Cluster', 'Total_Crime_Rate'], ascending=[True, False])
    
    # Add standardized features for visualization
    scaled_features = pd.DataFrame(
        X_scaled, 
        columns=X.columns, 
        index=X.index
    )
    
    return result, kmeans.cluster_centers_, scaled_features

def visualize_clusters(result, cluster_centers, scaled_features):
    """
    Create visualizations of the clustering results
    """
    # Plot heatmap of cluster centers
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        pd.DataFrame(cluster_centers, columns=scaled_features.columns),
        cmap='RdYlBu',
        center=0,
        annot=True,
        fmt='.2f'
    )
    plt.title('Cluster Centers Heatmap\n(Standardized Values)')
    plt.show()
    
    # Create cluster summary
    print("\nCluster Summary:")
    for cluster in result['Cluster'].unique():
        cluster_data = result[result['Cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Number of states: {len(cluster_data)}")
        print("States (sorted by total crime rate):")
        for _, row in cluster_data.iterrows():
            print(f"- {row['State']} (Total crime rate: {row['Total_Crime_Rate']:.2f})")
        print("\nDominant crime types:")
        cluster_center = pd.DataFrame(cluster_centers[cluster], 
                                    index=scaled_features.columns)
        dominant_crimes = cluster_center[cluster_center[0] > 0].index.tolist()
        print(", ".join(dominant_crimes) if dominant_crimes else "No significantly high crime rates")

def main():
    # Read and prepare the data
    df = pd.read_csv('CrimesOnWomenData.csv')
    state_crimes = prepare_data(df)
    
    # Find optimal number of clusters
    optimal_clusters = find_optimal_clusters(state_crimes)
    print(f"\nOptimal number of clusters: {optimal_clusters}")
    
    # Perform clustering
    result, cluster_centers, scaled_features = perform_clustering(
        state_crimes, 
        optimal_clusters
    )
    
    # Visualize results
    visualize_clusters(result, cluster_centers, scaled_features)
    
    return result

if __name__ == "__main__":
    result_df = main()