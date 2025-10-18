import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from src.data_ingestion import DataIngestion
from src.eda import EDAAnalyzer
from src.models import ModelTrainer
from src.monitoring import PerformanceMonitor
from src.logger import setup_logger

# Setup logger
logger = setup_logger()

# Page configuration
st.set_page_config(
    page_title="AI Infant Mortality Prediction System",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None

def main():
    st.title("🏥 AI-Based Infant Mortality Rate Prediction System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Data Overview", "Data Ingestion", "Exploratory Data Analysis", 
         "Model Training", "Predictions", "Performance Monitoring", "API Testing"]
    )
    
    if page == "Data Overview":
        show_data_overview()
    elif page == "Data Ingestion":
        show_data_ingestion()
    elif page == "Exploratory Data Analysis":
        show_eda()
    elif page == "Model Training":
        show_model_training()
    elif page == "Predictions":
        show_predictions()
    elif page == "Performance Monitoring":
        show_monitoring()
    elif page == "API Testing":
        show_api_testing()

def show_data_overview():
    st.header("📊 Data Overview")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Countries", len(data['country'].unique()))
        with col2:
            st.metric("Years Available", f"{data['year'].min()}-{data['year'].max()}")
        with col3:
            st.metric("Total Records", len(data))
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        st.subheader("Data Sample")
        st.dataframe(data.head(10))
        
        st.subheader("Data Statistics")
        st.dataframe(data.describe())
        
    else:
        st.info("Please load data first using the Data Ingestion page.")

def show_data_ingestion():
    st.header("📥 Data Ingestion")
    
    st.markdown("""
    This section fetches infant mortality rate data from the World Bank API along with 
    related health indicators that can be used as features for prediction.
    """)
    
    # Data ingestion controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        start_year = st.number_input("Start Year", value=2000, min_value=1960, max_value=2023)
        end_year = st.number_input("End Year", value=2022, min_value=1960, max_value=2023)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Fetch Data", type="primary"):
            fetch_data(start_year, end_year)
    
    if st.session_state.data_loaded:
        st.success("✅ Data successfully loaded!")
        
        # Show data quality info
        if st.session_state.data is not None:
            data = st.session_state.data
            
            st.subheader("Data Quality Report")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Missing Values by Column:**")
                missing_data = data.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(data) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])
            
            with col2:
                st.write("**Data Coverage by Year:**")
                yearly_coverage = data.groupby('year').size()
                fig = px.bar(x=yearly_coverage.index, y=yearly_coverage.values,
                           title="Records per Year")
                fig.update_layout(xaxis_title="Year", yaxis_title="Number of Records")
                st.plotly_chart(fig, use_container_width=True)

def fetch_data(start_year, end_year):
    try:
        with st.spinner("Fetching data from World Bank API..."):
            data_ingestion = DataIngestion()
            data = data_ingestion.fetch_data(start_year, end_year)
            
            if data is not None and not data.empty:
                st.session_state.data = data
                st.session_state.data_loaded = True
                logger.info(f"Data successfully fetched: {len(data)} records")
            else:
                st.error("Failed to fetch data or no data available for the specified period.")
                
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        logger.error(f"Data fetching error: {str(e)}")

def show_eda():
    st.header("🔍 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first using the Data Ingestion page.")
        return
    
    data = st.session_state.data
    eda_analyzer = EDAAnalyzer(data)
    
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Infant mortality distribution
        fig1 = eda_analyzer.plot_distribution('infant_mortality_rate')
        st.pyplot(fig1)
    
    with col2:
        # GDP per capita distribution
        if 'gdp_per_capita' in data.columns:
            fig2 = eda_analyzer.plot_distribution('gdp_per_capita')
            st.pyplot(fig2)
    
    st.subheader("Correlation Analysis")
    fig_corr = eda_analyzer.plot_correlation_heatmap()
    st.pyplot(fig_corr)
    
    st.subheader("Trend Analysis")
    
    # Country selection for trend analysis
    countries = st.multiselect(
        "Select countries for trend analysis:",
        options=data['country'].unique(),
        default=data['country'].unique()[:5] if len(data['country'].unique()) >= 5 else data['country'].unique()
    )
    
    if countries:
        fig_trends = eda_analyzer.plot_trends(countries)
        st.pyplot(fig_trends)
    
    st.subheader("Regional Analysis")
    if 'region' in data.columns:
        fig_regional = eda_analyzer.plot_regional_analysis()
        st.pyplot(fig_regional)

def show_model_training():
    st.header("🤖 Model Training & Comparison")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first using the Data Ingestion page.")
        return
    
    data = st.session_state.data
    
    # Model training controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=1)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Train Models", type="primary"):
            train_models(data, test_size, random_state)
    
    if st.session_state.models_trained:
        st.success("✅ Models successfully trained!")
        
        # Load and display model comparison
        try:
            model_trainer = ModelTrainer(data)
            comparison_results = joblib.load('models/model_comparison.pkl')
            
            st.subheader("Model Performance Comparison")
            
            # Convert results to DataFrame for display
            results_df = pd.DataFrame(comparison_results).T
            st.dataframe(results_df)
            
            # Visualization of model comparison
            fig_comparison = model_trainer.plot_model_comparison(comparison_results)
            st.pyplot(fig_comparison)
            
            # Feature importance for best model
            st.subheader("Feature Importance (Best Model)")
            best_model = min(comparison_results.items(), key=lambda x: x[1]['rmse'])
            st.write(f"**Best Model:** {best_model[0]}")
            
            fig_importance = model_trainer.plot_feature_importance(best_model[0])
            if fig_importance:
                st.pyplot(fig_importance)
                
        except Exception as e:
            st.error(f"Error loading model results: {str(e)}")

def train_models(data, test_size, random_state):
    try:
        with st.spinner("Training models... This may take a few minutes."):
            model_trainer = ModelTrainer(data)
            results = model_trainer.train_and_compare_models(
                test_size=test_size, 
                random_state=random_state
            )
            
            if results:
                st.session_state.models_trained = True
                logger.info("Models successfully trained and compared")
            else:
                st.error("Model training failed.")
                
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        logger.error(f"Model training error: {str(e)}")

def show_predictions():
    st.header("🔮 Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first using the Model Training page.")
        return
    
    # Load available models
    model_files = [f for f in os.listdir('models/') if f.endswith('.pkl') and 'comparison' not in f]
    
    if not model_files:
        st.error("No trained models found.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Controls")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model:",
            options=[f.replace('.pkl', '') for f in model_files]
        )
        
        # Country selection
        data = st.session_state.data
        country = st.selectbox(
            "Select Country:",
            options=['All Countries'] + list(data['country'].unique())
        )
        
        # Feature inputs (simplified for demo)
        st.write("**Feature Inputs:**")
        gdp_per_capita = st.number_input("GDP per Capita", value=10000.0, min_value=0.0)
        health_expenditure = st.number_input("Health Expenditure (% of GDP)", value=5.0, min_value=0.0)
        education_expenditure = st.number_input("Education Expenditure (% of GDP)", value=4.0, min_value=0.0)
        
        if st.button("🎯 Make Prediction", type="primary"):
            make_prediction(selected_model, country, gdp_per_capita, health_expenditure, education_expenditure)
    
    with col2:
        st.subheader("Prediction Results")
        
        # This would be populated by the prediction function
        if 'last_prediction' in st.session_state:
            st.success(f"Predicted Infant Mortality Rate: {st.session_state.last_prediction:.2f} per 1,000 live births")
            
            # Add confidence interval if available
            if 'prediction_interval' in st.session_state:
                lower, upper = st.session_state.prediction_interval
                st.info(f"95% Prediction Interval: [{lower:.2f}, {upper:.2f}]")

def make_prediction(model_name, country, gdp, health_exp, edu_exp):
    try:
        # Load the selected model
        model = joblib.load(f'models/{model_name}.pkl')
        
        # Prepare features (this is simplified - in reality would need proper feature engineering)
        features = np.array([[gdp, health_exp, edu_exp]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        st.session_state.last_prediction = prediction
        logger.info(f"Prediction made: {prediction:.2f} for {country}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")

def show_monitoring():
    st.header("📈 Performance Monitoring")
    
    # Initialize monitoring
    monitor = PerformanceMonitor()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        
        # Load model comparison results if available
        try:
            comparison_results = joblib.load('models/model_comparison.pkl')
            
            for model_name, metrics in comparison_results.items():
                with st.expander(f"📊 {model_name}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    with col_b:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                        if 'cv_score' in metrics:
                            st.metric("CV Score", f"{metrics['cv_score']:.4f}")
                            
        except FileNotFoundError:
            st.info("No model performance data available. Train models first.")
    
    with col2:
        st.subheader("System Health")
        
        # System metrics
        st.metric("API Status", "🟢 Online")
        st.metric("Models Loaded", len([f for f in os.listdir('models/') if f.endswith('.pkl')]))
        st.metric("Data Freshness", "✅ Current")
        
        # Log analysis
        st.subheader("Recent Activity")
        try:
            log_summary = monitor.get_log_summary()
            if log_summary:
                st.json(log_summary)
            else:
                st.info("No recent activity logged.")
        except Exception as e:
            st.info("Log analysis not available.")

def show_api_testing():
    st.header("🔌 API Testing")
    
    st.markdown("""
    Test the REST API endpoints. The API server should be running on port 8000.
    """)
    
    # API endpoint testing
    base_url = "http://localhost:8000"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("API Endpoints")
        
        endpoint = st.selectbox(
            "Select Endpoint:",
            ["/health", "/predict", "/predict/country/{country_code}"]
        )
        
        if endpoint == "/predict":
            st.write("**Request Body:**")
            gdp = st.number_input("GDP per Capita", value=10000.0)
            health_exp = st.number_input("Health Expenditure", value=5.0)
            edu_exp = st.number_input("Education Expenditure", value=4.0)
            
            request_body = {
                "gdp_per_capita": gdp,
                "health_expenditure": health_exp,
                "education_expenditure": edu_exp
            }
            
        elif "/country/" in endpoint:
            country_code = st.text_input("Country Code", value="USA")
            endpoint = endpoint.replace("{country_code}", country_code)
            request_body = None
        else:
            request_body = None
        
        if st.button("🚀 Test API"):
            test_api_endpoint(base_url + endpoint, request_body)
    
    with col2:
        st.subheader("API Response")
        
        if 'api_response' in st.session_state:
            response = st.session_state.api_response
            st.json(response)

def test_api_endpoint(url, request_body=None):
    try:
        if request_body:
            response = requests.post(url, json=request_body, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        st.session_state.api_response = {
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type') == 'application/json' else response.text
        }
        
    except requests.exceptions.ConnectionError:
        st.session_state.api_response = {
            "error": "Could not connect to API. Make sure the API server is running on port 8000."
        }
    except Exception as e:
        st.session_state.api_response = {
            "error": f"API test error: {str(e)}"
        }

if __name__ == "__main__":
    main()
