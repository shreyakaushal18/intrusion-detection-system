import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import urllib.request
import os

# Page configuration
st.set_page_config(
    page_title="KDD Cup 99 ML Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Network Intrusion Detection System")
st.markdown("**Machine Learning Model with Performance Metrics**")
st.markdown("---")

# Column names for KDD Cup 99 dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

@st.cache_data
def load_data():
    """Load KDD Cup 99 dataset"""
    data_file = 'kddcup_data.csv'
    
    if not os.path.exists(data_file):
        with st.spinner('Downloading KDD Cup 99 dataset... This may take a moment.'):
            try:
                # Download a sample of the dataset (10% version for faster processing)
                url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
                urllib.request.urlretrieve(url, 'kddcup.data_10_percent.gz')
                
                # Load the data
                df = pd.read_csv('kddcup.data_10_percent.gz', names=column_names, compression='gzip')
                df.to_csv(data_file, index=False)
            except Exception as e:
                st.error(f"Error downloading dataset: {e}")
                st.info("Creating sample dataset for demonstration...")
                # Create a small sample dataset for demo purposes
                np.random.seed(42)
                df = pd.DataFrame({
                    col: np.random.randn(1000) if col not in ['protocol_type', 'service', 'flag', 'label'] 
                    else np.random.choice(['tcp', 'udp', 'icmp'], 1000) if col == 'protocol_type'
                    else np.random.choice(['http', 'smtp', 'ftp'], 1000) if col == 'service'
                    else np.random.choice(['SF', 'S0', 'REJ'], 1000) if col == 'flag'
                    else np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r'], 1000)
                    for col in column_names
                })
                df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)
    
    return df

@st.cache_data
def preprocess_data(df, sample_size=50000):
    """Preprocess the dataset"""
    # Sample data for faster processing if dataset is large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Remove the trailing dot from labels if present
    df['label'] = df['label'].str.replace('.', '', regex=False)
    
    # Binary classification: normal vs attack
    df['attack'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Encode categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

# Sidebar controls
st.sidebar.header("Model Configuration")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)
n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Max Depth", 5, 50, 20, 5)

# Load and preprocess data
with st.spinner('Loading data...'):
    df = load_data()
    df_processed, label_encoders = preprocess_data(df)

# Display dataset info
st.header("Dataset Overview(KDD Cup 99)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", len(df_processed))
with col2:
    st.metric("Features", len(df_processed.columns) - 2)
with col3:
    st.metric("Normal Traffic", len(df_processed[df_processed['attack'] == 0]))
with col4:
    st.metric("Attack Traffic", len(df_processed[df_processed['attack'] == 1]))

# Show sample data
with st.expander("View Sample Data"):
    st.dataframe(df_processed.head(10))

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    # Attack distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    attack_counts = df_processed['attack'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(attack_counts, labels=['Normal', 'Attack'], autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax.set_title('Distribution of Traffic (Normal vs Attack)', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col2:
    # Label distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    label_counts = df_processed['label'].value_counts().head(10)
    sns.barplot(x=label_counts.values, y=label_counts.index, palette='viridis', ax=ax)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Attack Type', fontsize=12)
    ax.set_title('Top 10 Attack Types Distribution', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# Feature correlation heatmap
st.subheader("Feature Correlation Heatmap")
numeric_features = df_processed.select_dtypes(include=[np.number]).columns[:15]
fig, ax = plt.subplots(figsize=(6, 4))
correlation_matrix = df_processed[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
st.pyplot(fig)
plt.close()

# Model Training
st.header("Model Training & Evaluation")

if st.button("Train Model", type="primary"):
    with st.spinner('Training model... Please wait.'):
        # Prepare features and target
        X = df_processed.drop(['label', 'attack'], axis=1)
        y = df_processed['attack']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Store results in session state
        st.session_state['model_trained'] = True
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['model'] = model
        st.session_state['feature_names'] = X.columns
        
        st.success("Model trained successfully!")
        st.rerun()

# Display results if model is trained
if st.session_state.get('model_trained', False):
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']
    model = st.session_state['model']
    feature_names = st.session_state['feature_names']
    
    st.subheader("Performance Metrics")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
    with col3:
        st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
    with col4:
        st.metric("F1 Score", f"{f1:.4f}", f"{f1*100:.2f}%")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        # Display confusion matrix values
        st.write("**Matrix Values:**")
        st.write(f"- True Negatives (TN): {cm[0][0]}")
        st.write(f"- False Positives (FP): {cm[0][1]}")
        st.write(f"- False Negatives (FN): {cm[1][0]}")
        st.write(f"- True Positives (TP): {cm[1][1]}")
    
    with col2:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=feature_importance, x='importance', y='feature', 
                   palette='rocket', ax=ax)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # Classification Report
    st.subheader("Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))

else:
    st.info("Click the 'Train Model' button above to start training!")

# Footer
st.markdown("---")
st.markdown("**Built with Streamlit | Using KDD Cup 99 Dataset for Network Intrusion Detection**")
