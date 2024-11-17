import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from medical_extraction import initialize_processor, analyze_medical_symptoms
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class CrimeQueryProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.crime_keywords = {
            'kidnapping': ['kidnap', 'abduct', 'kidnapping', 'abduction'],
            'rape': ['rape', 'sexual assault', 'sexual violence'],
            'domestic_violence': ['domestic violence', 'domestic abuse', 'household abuse'],
            'harassment': ['harassment', 'stalking', 'intimidation'],
            'dowry': ['dowry death', 'dowry harassment', 'dowry'],
            'trafficking': ['trafficking', 'human trafficking'],
            'total_crime': ['total crime', 'overall crime', 'unsafe', 'dangerous']
        }
        self.state_synonyms = {
            'delhi': ['delhi', 'new delhi', 'dilli'],
            'mumbai': ['mumbai', 'maharashtra', 'bombay'],
            'up': ['uttar pradesh', 'up'],
            # Add more state synonyms as needed
        }
        
    def preprocess_text(self, text):
        """Preprocess the query text."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
        
    def extract_query_parameters(self, query):
        """Extract relevant parameters from the query."""
        processed_query = self.preprocess_text(query)
        
        # Initialize parameters
        params = {
            'crime_type': None,
            'year': None,
            'state': None,
            'query_type': None,
            'comparison': False,
            'trend': False
        }
        
        # Extract year
        year_pattern = r'\b(19|20)\d{2}\b'
        year_match = re.search(year_pattern, query)
        if year_match:
            params['year'] = int(year_match.group())
            
        # Detect query type and crime type
        words = processed_query.split()
        
        # Check for trends
        trend_keywords = ['trend', 'increase', 'decrease', 'change', 'growing', 'rising']
        if any(word in words for word in trend_keywords):
            params['trend'] = True
            
        # Check for comparisons
        comparison_keywords = ['compare', 'comparison', 'versus', 'vs', 'against']
        if any(word in words for word in comparison_keywords):
            params['comparison'] = True
            
        # Detect crime type
        for crime_type, keywords in self.crime_keywords.items():
            if any(keyword in processed_query for keyword in keywords):
                params['crime_type'] = crime_type
                break
                
        # Detect query type
        if 'most' in words or 'highest' in words or 'worst' in words:
            params['query_type'] = 'highest'
        elif 'least' in words or 'lowest' in words or 'safest' in words:
            params['query_type'] = 'lowest'
            
        return params

class CrimeAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.query_processor = CrimeQueryProcessor()
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess the crime data."""
        self.df.fillna(0, inplace=True)
        self.crime_matrix = self.df.iloc[:, 2:]
        self.crime_matrix_scaled = StandardScaler().fit_transform(self.crime_matrix)
        
    def process_query(self, query):
        """Process natural language query and return relevant information."""
        params = self.query_processor.extract_query_parameters(query)
        
        try:
            results = {}
            
            if params['trend']:
                results = self.analyze_trends(params['crime_type'])
            elif params['comparison']:
                results = self.compare_states(params['crime_type'], params['year'])
            else:
                results = self.get_crime_statistics(params)
                
            return self.format_results(results, params)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
            
    def analyze_trends(self, crime_type):
        """Analyze trends for a specific crime type."""
        if crime_type:
            yearly_data = self.df.groupby('Year')[crime_type].sum()
            trend_change = (yearly_data.iloc[-1] - yearly_data.iloc[0]) / yearly_data.iloc[0] * 100
            
            return {
                'trend_data': yearly_data,
                'total_change': trend_change,
                'highest_year': yearly_data.idxmax(),
                'lowest_year': yearly_data.idxmin()
            }
            
    def compare_states(self, crime_type, year=None):
        """Compare states for a specific crime type."""
        if not year:
            year = self.df['Year'].max()
            
        data = self.df[self.df['Year'] == year]
        if crime_type:
            return data[['State', crime_type]].sort_values(by=crime_type, ascending=False)
            
    def get_crime_statistics(self, params):
        """Get crime statistics based on query parameters."""
        if params['year']:
            data = self.df[self.df['Year'] == params['year']]
        else:
            data = self.df[self.df['Year'] == self.df['Year'].max()]
            
        if params['crime_type']:
            sorted_data = data.sort_values(by=params['crime_type'], 
                                         ascending=params['query_type'] == 'lowest')
            return {
                'state_data': sorted_data[['State', params['crime_type']]].head(5),
                'total': sorted_data[params['crime_type']].sum(),
                'avg': sorted_data[params['crime_type']].mean()
            }
            
    def format_results(self, results, params):
        """Format the results for display."""
        if not results:
            return "I couldn't find relevant information for your query."
            
        if params['trend']:
            fig, ax = plt.subplots(figsize=(10, 6))
            results['trend_data'].plot(kind='line', ax=ax)
            plt.title(f"Trend Analysis: {params['crime_type']}")
            plt.xlabel("Year")
            plt.ylabel("Number of Cases")
            
            return {
                'text': f"""
                Trend Analysis for {params['crime_type']}:
                - Total change: {results['total_change']:.2f}%
                - Highest in: {results['highest_year']}
                - Lowest in: {results['lowest_year']}
                """,
                'plot': fig
            }
            
        elif isinstance(results, dict) and 'state_data' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=results['state_data'], 
                       x='State', 
                       y=params['crime_type'])
            plt.xticks(rotation=45)
            plt.title(f"{params['crime_type']} Cases by State")
            
            return {
                'text': f"""
                Statistics for {params['crime_type']}:
                - Total cases: {results['total']:,.0f}
                - Average cases per state: {results['avg']:,.0f}
                """,
                'plot': fig,
                'data': results['state_data']
            }
            
        return results

def main():
    st.set_page_config(layout="wide", page_title="Women Safety Analysis System")
    
    st.sidebar.title("Analysis Options")
    analysis_type = st.sidebar.radio("Select Analysis Type", 
                                   ["Women Crime Report", "Medical Disorder Extraction"])
    
    if analysis_type == "Women Crime Report":
        st.title("Women Crime Analysis Dashboard")
        
        try:
            crime_analysis = CrimeAnalysis('CrimesOnWomenData.csv')
            
            # Query Section
            st.subheader("Crime Data Query")
            query = st.text_area(
                "Ask your question about women's safety and crime statistics:",
                placeholder="Example: Which state had the highest kidnapping cases in 2020?"
            )
            
            if st.button("Analyze"):
                if query:
                    results = crime_analysis.process_query(query)
                    
                    if isinstance(results, dict):
                        if 'text' in results:
                            st.write(results['text'])
                        if 'plot' in results:
                            st.pyplot(results['plot'])
                        if 'data' in results:
                            st.dataframe(results['data'])
                    else:
                        st.write(results)
                else:
                    st.warning("Please enter your query first.")
                    
            # Display some example queries
            with st.expander("Example Queries"):
                st.write("""
                - Which state has the highest number of rape cases?
                - Show me the trend of domestic violence over the years
                - Compare kidnapping cases across states in 2020
                - What is the safest state for women?
                - Which states have the highest total crime against women?
                """)
                
        except Exception as e:
            st.error(f"Error in crime analysis: {e}")
            
    else:  # Medical Disorder Extraction
        # [Previous medical analysis code remains the same]
        pass

if __name__ == "__main__":
    main()
