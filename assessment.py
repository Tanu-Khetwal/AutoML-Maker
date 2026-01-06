import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

st.set_page_config(page_title="Manual AutoML App", layout="wide")

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def build_preprocessor(numeric_features, categorical_features):
    st.info(f"Using Numeric features: {list(numeric_features)}")
    st.info(f"Using Categorical features: {list(categorical_features)}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' 
    )
    return preprocessor

def get_models(problem_type):
    if problem_type == "Classification":
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC()
        }
    else: 
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Support Vector Machine': SVR()
        }

st.title("🔧 Build Your Own ML Pipeline")
st.write("Upload data, create new features, train models, save the best one, and test it.")

if 'results' not in st.session_state:
    st.session_state.results = None
if 'pipelines' not in st.session_state:
    st.session_state.pipelines = {}
if 'df' not in st.session_state:
    st.session_state.df = None


with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None and st.session_state.df is None:
        df_loaded = load_data(uploaded_file)
        if df_loaded is not None:
            st.session_state.df = df_loaded
    
    if st.session_state.df is not None:
        st.success("Data loaded successfully!")
        st.dataframe(st.session_state.df.head(), height=150)
    

        all_columns = st.session_state.df.columns.tolist()

        st.header("2. Define Features (X)")
        st.info("Select which columns you want to use as features.")
        
        numeric_cols = st.multiselect(
            "Select NUMERIC features", 
            all_columns, 
            help="Select columns with numbers (e.g., age, price)."
        )
        
        categorical_cols = st.multiselect(
            "Select CATEGORICAL features", 
            all_columns,
            help="Select columns with text or categories (e.g., city, gender)."
        )
        
        st.header("3. Define Target (y)")
        st.info("Select the single column you want to predict.")
        
        target_col = st.selectbox(
            "Select your TARGET variable", 
            all_columns
        )
        
        problem_type = st.selectbox(
            "Select problem type", 
            ["Classification", "Regression"]
        )
        
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols
        st.session_state.target_col = target_col
        st.session_state.problem_type = problem_type
        
        
        st.header("4. Run Training")
        if st.button("Run Model Training", use_container_width=True, type="primary"):
            if not numeric_cols and not categorical_cols:
                st.error("Please select at least one feature column (numeric or categorical).")
            elif target_col is None:
                st.error("Please select a target variable.")
            elif target_col in numeric_cols or target_col in categorical_cols:
                 st.error("Target column cannot also be a feature. Please unselect it from the feature lists.")
            else:
                with st.spinner("Preparing data and training models..."):
                    try:
                        df_clean = st.session_state.df.copy()
                        numeric_cols = st.session_state.numeric_cols
                        categorical_cols = st.session_state.categorical_cols
                        target_col = st.session_state.target_col
                        problem_type = st.session_state.problem_type
                        
                        st.write("Cleaning numeric columns...")
                        for col in numeric_cols:
                            if col in df_clean.columns and df_clean[col].dtype == 'object':
                                df_clean[col] = pd.to_numeric(
                                    df_clean[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                                    errors='coerce'
                                )
                        
                        st.write("Cleaning target column...")
                        if target_col in df_clean.columns and df_clean[target_col].dtype == 'object':
                            df_clean[target_col] = pd.to_numeric(
                                df_clean[target_col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                                errors='coerce'
                            )
                        
                        used_cols = numeric_cols + categorical_cols + [target_col]
                        df_clean.dropna(subset=used_cols, inplace=True)
                        st.write("Cleaning and NaN removal complete.")
                        
                        if df_clean.empty:
                            st.error("The dataset is empty after cleaning. Please check your data.")
                        else:
                            feature_cols = numeric_cols + categorical_cols
                            X = df_clean[feature_cols]
                            y = df_clean[target_col]
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            preprocessor = build_preprocessor(numeric_cols, categorical_cols)
                            models = get_models(problem_type)
                            
                            results = {}
                            pipelines = {}

                            for name, model in models.items():
                                pipeline = Pipeline(steps=[
                                    ('preprocessor', preprocessor),
                                    ('model', model)
                                ])
                                pipeline.fit(X_train, y_train)
                                preds = pipeline.predict(X_test)
                                
                                if problem_type == "Classification":
                                    score = accuracy_score(y_test, preds)
                                else:
                                    score = r2_score(y_test, preds)
                                
                                results[name] = score
                                pipelines[name] = pipeline 
                            
                            st.session_state.results = results
                            st.session_state.pipelines = pipelines
                            st.success("Model training complete!")

                    except Exception as e:
                        st.error(f"An error occurred during training: {e}")
                        st.session_state.results = None

if st.session_state.results is not None:
    st.header("5. View Model Comparison")
    
    metric_name = "Accuracy" if st.session_state.problem_type == "Classification" else "R-squared"
    results_df = pd.DataFrame.from_dict(st.session_state.results, orient='index', columns=[metric_name])
    results_df = results_df.sort_values(by=metric_name, ascending=False)
    st.dataframe(results_df)

    st.header("6. Save Your Chosen Model")
    st.write("Select the model you want to save")

    model_to_save_name = st.selectbox("Select model to save:", results_df.index)
    
    if st.button(f"Save '{model_to_save_name}' Model", use_container_width=True, type="primary"):
        with st.spinner(f"Saving '{model_to_save_name}'..."):
            try:
                pipeline_to_save = st.session_state.pipelines[model_to_save_name]
                model_filename = "saved_pipeline.joblib"
                
                joblib.dump(pipeline_to_save, model_filename)
                
                st.success(f"Model saved as '{model_filename}' in your app folder.")

                with open(model_filename, "rb") as f:
                    st.download_button(
                        label="Download Saved Model Pipeline",
                        data=f,
                        file_name=model_filename,
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"An error occurred while saving the model: {e}")

    st.header("7. Make a Prediction")
    st.write(f"Test the model you selected above ('{model_to_save_name}') with new data.")

    numeric_cols = st.session_state.numeric_cols
    categorical_cols = st.session_state.categorical_cols
    
    with st.form("prediction_form"):
        input_data = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numeric Features")
            for col in numeric_cols:
                input_data[col] = st.number_input(f"Enter {col}", value=0.0, format="%.2f")
                
        with col2:
            st.subheader("Categorical Features")
            for col in categorical_cols:
                unique_values = st.session_state.df[col].unique().tolist()
                input_data[col] = st.selectbox(f"Select {col}", options=unique_values)
        
        submit_button = st.form_submit_button("Get Prediction")

    if submit_button:
        try:
            pipeline = st.session_state.pipelines[model_to_save_name]
            
            input_df = pd.DataFrame([input_data]) 
            
            prediction = pipeline.predict(input_df)[0]
            
            problem_type = st.session_state.problem_type
            if problem_type == "Classification":
                st.success(f"**Predicted {st.session_state.target_col}:** `{prediction}`")
            else: 
                st.success(f"**Predicted {st.session_state.target_col}:** `{prediction:,.2f}`")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Please upload your data in the sidebar to begin.")