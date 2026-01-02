import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("CREDIT CARD DEFAULT PREDICTION - TRAINING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: DATA GENERATION (Replace with your actual credit data)
# ============================================================================

def generate_credit_data(n_samples=10000):
    """Generate synthetic credit card default data"""
    
    print("\n[1/7] Generating synthetic credit data...")
    
    data = {
        'customer_id': [f'CUST{i:06d}' for i in range(n_samples)],
        'credit_limit': np.random.uniform(1000, 50000, n_samples),
        'age': np.random.randint(21, 70, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        
        # Payment history (last 6 months: -1=delay, 0=paid, 1=no use)
        'pay_status_1': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2]),
        'pay_status_2': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2]),
        'pay_status_3': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2]),
        'pay_status_4': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2]),
        'pay_status_5': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2]),
        'pay_status_6': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2]),
        
        # Bill amounts (last 6 months)
        'bill_amt_1': np.random.uniform(0, 30000, n_samples),
        'bill_amt_2': np.random.uniform(0, 30000, n_samples),
        'bill_amt_3': np.random.uniform(0, 30000, n_samples),
        'bill_amt_4': np.random.uniform(0, 30000, n_samples),
        'bill_amt_5': np.random.uniform(0, 30000, n_samples),
        'bill_amt_6': np.random.uniform(0, 30000, n_samples),
        
        # Payment amounts (last 6 months)
        'pay_amt_1': np.random.uniform(0, 10000, n_samples),
        'pay_amt_2': np.random.uniform(0, 10000, n_samples),
        'pay_amt_3': np.random.uniform(0, 10000, n_samples),
        'pay_amt_4': np.random.uniform(0, 10000, n_samples),
        'pay_amt_5': np.random.uniform(0, 10000, n_samples),
        'pay_amt_6': np.random.uniform(0, 10000, n_samples),
        
        # Account metrics
        'account_age_months': np.random.randint(6, 120, n_samples),
        'num_bank_accounts': np.random.randint(1, 5, n_samples),
        'num_credit_cards': np.random.randint(1, 8, n_samples),
        'avg_monthly_income': np.random.uniform(2000, 15000, n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic default probability
    default_prob = (
        (df[['pay_status_1', 'pay_status_2', 'pay_status_3']].sum(axis=1) < -1) * 0.3 +
        (df['bill_amt_1'] / df['credit_limit'] > 0.8) * 0.25 +
        (df['pay_amt_1'] / (df['bill_amt_1'] + 1) < 0.1) * 0.2 +
        (df['employment_status'] == 'Unemployed') * 0.15 +
        (df['age'] < 25) * 0.1 +
        np.random.uniform(0, 0.15, n_samples)
    )
    
    df['default'] = (default_prob > 0.45).astype(int)
    
    print(f"✓ Generated {n_samples} records")
    print(f"✓ Default rate: {df['default'].mean()*100:.2f}%")
    
    return df

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df):
    """Comprehensive EDA with visualizations"""
    
    print("\n[2/7] Performing Exploratory Data Analysis...")
    
    print("\n" + "="*80)
    print("DATA OVERVIEW")
    print("="*80)
    
    print(f"\nShape: {df.shape}")
    print(f"Default Rate: {df['default'].mean()*100:.2f}%")
    print(f"\nMissing Values:\n{df.isnull().sum().sum()} total")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Default distribution
    ax1 = plt.subplot(3, 4, 1)
    df['default'].value_counts().plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Default Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Default Status')
    ax1.set_ylabel('Count')
    
    # 2. Age distribution
    ax2 = plt.subplot(3, 4, 2)
    df.boxplot(column='age', by='default', ax=ax2)
    ax2.set_title('Age by Default Status')
    plt.suptitle('')
    
    # 3. Credit limit distribution
    ax3 = plt.subplot(3, 4, 3)
    df.boxplot(column='credit_limit', by='default', ax=ax3)
    ax3.set_title('Credit Limit by Default')
    plt.suptitle('')
    
    # 4. Gender vs Default
    ax4 = plt.subplot(3, 4, 4)
    pd.crosstab(df['gender'], df['default'], normalize='index').plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Default Rate by Gender')
    ax4.legend(['No Default', 'Default'])
    
    # 5. Education vs Default
    ax5 = plt.subplot(3, 4, 5)
    pd.crosstab(df['education'], df['default'], normalize='index').plot(kind='bar', ax=ax5)
    ax5.set_title('Default Rate by Education')
    ax5.legend(['No Default', 'Default'])
    
    # 6. Marital Status vs Default
    ax6 = plt.subplot(3, 4, 6)
    pd.crosstab(df['marital_status'], df['default'], normalize='index').plot(kind='bar', ax=ax6)
    ax6.set_title('Default Rate by Marital Status')
    ax6.legend(['No Default', 'Default'])
    
    # 7. Employment vs Default
    ax7 = plt.subplot(3, 4, 7)
    pd.crosstab(df['employment_status'], df['default'], normalize='index').plot(kind='bar', ax=ax7)
    ax7.set_title('Default Rate by Employment')
    ax7.legend(['No Default', 'Default'])
    
    # 8. Payment status correlation
    ax8 = plt.subplot(3, 4, 8)
    pay_cols = ['pay_status_1', 'pay_status_2', 'pay_status_3']
    df[pay_cols].corrwith(df['default']).plot(kind='bar', ax=ax8, color='coral')
    ax8.set_title('Payment Status Correlation with Default')
    
    # 9. Bill amount trend
    ax9 = plt.subplot(3, 4, 9)
    bill_cols = ['bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6']
    df.groupby('default')[bill_cols].mean().T.plot(ax=ax9, marker='o')
    ax9.set_title('Average Bill Amount Trend')
    ax9.legend(['No Default', 'Default'])
    
    # 10. Payment amount trend
    ax10 = plt.subplot(3, 4, 10)
    pay_amt_cols = ['pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6']
    df.groupby('default')[pay_amt_cols].mean().T.plot(ax=ax10, marker='o')
    ax10.set_title('Average Payment Amount Trend')
    ax10.legend(['No Default', 'Default'])
    
    # 11. Correlation heatmap
    ax11 = plt.subplot(3, 4, 11)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', ax=ax11, cmap='coolwarm', cbar_kws={'shrink': 0.8})
    ax11.set_title('Feature Correlation Heatmap')
    
    # 12. Credit utilization
    ax12 = plt.subplot(3, 4, 12)
    df['utilization'] = df['bill_amt_1'] / df['credit_limit']
    df.boxplot(column='utilization', by='default', ax=ax12)
    ax12.set_title('Credit Utilization by Default')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('credit_eda_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ EDA visualizations saved as 'credit_eda_analysis.png'")
    plt.show()

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """Create advanced features for credit risk prediction"""
    
    print("\n[3/7] Engineering Features...")
    
    df_fe = df.copy()
    
    # 1. Credit utilization metrics
    df_fe['utilization_ratio'] = df_fe['bill_amt_1'] / (df_fe['credit_limit'] + 1)
    df_fe['avg_utilization'] = (df_fe['bill_amt_1'] + df_fe['bill_amt_2'] + df_fe['bill_amt_3']) / (3 * df_fe['credit_limit'] + 1)
    df_fe['max_utilization'] = df_fe[['bill_amt_1', 'bill_amt_2', 'bill_amt_3']].max(axis=1) / (df_fe['credit_limit'] + 1)
    
    # 2. Payment behavior features
    df_fe['total_delays'] = (df_fe[['pay_status_1', 'pay_status_2', 'pay_status_3', 
                                     'pay_status_4', 'pay_status_5', 'pay_status_6']] == -1).sum(axis=1)
    df_fe['recent_delays'] = (df_fe[['pay_status_1', 'pay_status_2', 'pay_status_3']] == -1).sum(axis=1)
    df_fe['payment_trend'] = df_fe['pay_status_1'] - df_fe['pay_status_6']
    
    # 3. Bill amount features
    df_fe['bill_increase'] = df_fe['bill_amt_1'] - df_fe['bill_amt_6']
    df_fe['bill_volatility'] = df_fe[['bill_amt_1', 'bill_amt_2', 'bill_amt_3', 
                                       'bill_amt_4', 'bill_amt_5', 'bill_amt_6']].std(axis=1)
    df_fe['avg_bill'] = df_fe[['bill_amt_1', 'bill_amt_2', 'bill_amt_3']].mean(axis=1)
    
    # 4. Payment amount features
    df_fe['payment_to_bill_ratio'] = df_fe['pay_amt_1'] / (df_fe['bill_amt_1'] + 1)
    df_fe['avg_payment'] = df_fe[['pay_amt_1', 'pay_amt_2', 'pay_amt_3']].mean(axis=1)
    df_fe['payment_decrease'] = df_fe['pay_amt_6'] - df_fe['pay_amt_1']
    
    # 5. Financial health indicators
    df_fe['debt_to_income'] = df_fe['bill_amt_1'] / (df_fe['avg_monthly_income'] + 1)
    df_fe['credit_per_card'] = df_fe['credit_limit'] / df_fe['num_credit_cards']
    df_fe['income_to_limit'] = df_fe['avg_monthly_income'] / (df_fe['credit_limit'] + 1)
    
    # 6. Risk flags
    df_fe['high_utilization'] = (df_fe['utilization_ratio'] > 0.8).astype(int)
    df_fe['consistent_delays'] = (df_fe['total_delays'] > 3).astype(int)
    df_fe['low_payment'] = (df_fe['payment_to_bill_ratio'] < 0.1).astype(int)
    df_fe['young_age'] = (df_fe['age'] < 25).astype(int)
    df_fe['unemployed'] = (df_fe['employment_status'] == 'Unemployed').astype(int)
    
    # 7. Account maturity
    df_fe['account_maturity_years'] = df_fe['account_age_months'] / 12
    df_fe['is_new_account'] = (df_fe['account_age_months'] < 12).astype(int)
    
    print(f"✓ Created {len(df_fe.columns) - len(df.columns)} new features")
    print(f"✓ Total features: {len(df_fe.columns)}")
    
    return df_fe

# ============================================================================
# STEP 4: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Prepare data for modeling"""
    
    print("\n[4/7] Preprocessing Data...")
    
    df_processed = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_marital = LabelEncoder()
    le_employment = LabelEncoder()
    
    df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])
    df_processed['education_encoded'] = le_education.fit_transform(df_processed['education'])
    df_processed['marital_encoded'] = le_marital.fit_transform(df_processed['marital_status'])
    df_processed['employment_encoded'] = le_employment.fit_transform(df_processed['employment_status'])
    
    # Select features
    feature_cols = [
        # Demographics
        'age', 'gender_encoded', 'education_encoded', 'marital_encoded', 
        'employment_encoded', 'num_dependents',
        
        # Account info
        'credit_limit', 'account_age_months', 'num_bank_accounts', 
        'num_credit_cards', 'avg_monthly_income',
        
        # Payment status
        'pay_status_1', 'pay_status_2', 'pay_status_3', 
        'pay_status_4', 'pay_status_5', 'pay_status_6',
        
        # Bill amounts
        'bill_amt_1', 'bill_amt_2', 'bill_amt_3',
        
        # Payment amounts
        'pay_amt_1', 'pay_amt_2', 'pay_amt_3',
        
        # Engineered features
        'utilization_ratio', 'avg_utilization', 'max_utilization',
        'total_delays', 'recent_delays', 'payment_trend',
        'bill_increase', 'bill_volatility', 'avg_bill',
        'payment_to_bill_ratio', 'avg_payment', 'payment_decrease',
        'debt_to_income', 'credit_per_card', 'income_to_limit',
        'high_utilization', 'consistent_delays', 'low_payment',
        'young_age', 'unemployed', 'account_maturity_years', 'is_new_account'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalanced data with SMOTE
    print(f"\n✓ Original training set: {X_train.shape}")
    print(f"  Class distribution: {dict(pd.Series(y_train).value_counts())}")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\n✓ After SMOTE: {X_train_balanced.shape}")
    print(f"  Class distribution: {dict(pd.Series(y_train_balanced).value_counts())}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Test set: {X_test.shape}")
    print(f"✓ Features used: {len(feature_cols)}")
    
    encoders = {
        'gender': le_gender,
        'education': le_education,
        'marital': le_marital,
        'employment': le_employment
    }
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, feature_cols, encoders

# ============================================================================
# STEP 5: MACHINE LEARNING MODELS
# ============================================================================

def train_ml_models(X_train, X_test, y_train, y_test):
    """Train traditional ML models"""
    
    print("\n[5/7] Training Machine Learning Models...")
    print("="*80)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba
        }
    
    return results

# ============================================================================
# STEP 6: DEEP LEARNING MODEL
# ============================================================================

def build_deep_learning_model(input_dim):
    """Build Deep Neural Network with BatchNorm and Dropout"""
    
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 1
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 2
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 3
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_deep_learning_model(X_train, X_test, y_train, y_test):
    """Train Deep Neural Network"""
    
    print("\n[6/7] Training Deep Learning Model...")
    print("="*80)
    
    # Build model
    model = build_deep_learning_model(X_train.shape[1])
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    # Train
    print("\nTraining Deep Neural Network...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n--- Deep Neural Network ---")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['auc'], label='Training AUC')
    axes[1].plot(history.history['val_auc'], label='Validation AUC')
    axes[1].set_title('Model AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dnn_training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history saved as 'dnn_training_history.png'")
    plt.show()
    
    return model, roc_auc, y_pred_proba

# ============================================================================
# STEP 7: MODEL COMPARISON & SAVING
# ============================================================================

def compare_and_save_models(ml_results, dnn_model, dnn_auc, dnn_proba, y_test, 
                           scaler, feature_cols, encoders):
    """Compare all models and save the best one"""
    
    print("\n[7/7] Comparing Models and Saving...")
    print("="*80)
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot ML models
    for name, result in ml_results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)
    
    # Plot DNN
    fpr, tpr, _ = roc_curve(y_test, dnn_proba)
    plt.plot(fpr, tpr, label=f"Deep Neural Network (AUC={dnn_auc:.3f})", 
             linewidth=3, linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('model_comparison_roc.png', dpi=300, bbox_inches='tight')
    print("\n✓ ROC comparison saved as 'model_comparison_roc.png'")
    plt.show()
    
    # Find best model
    all_scores = {name: result['roc_auc'] for name, result in ml_results.items()}
    all_scores['Deep Neural Network'] = dnn_auc
    
    best_model_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model_name]
    
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE SUMMARY")
    print('='*80)
    for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:30s}: {score:.4f}")
    print('='*80)
    print(f"\n🏆 Best Model: {best_model_name} with ROC-AUC: {best_score:.4f}")
    
    # Save models
    import pickle
    
    # Save best ML model
    if best_model_name != 'Deep Neural Network':
        best_ml_model = ml_results[best_model_name]['model']
    else:
        best_ml_model = ml_results['Gradient Boosting']['model']  # Save best ML as backup
    
    ml_artifacts = {
        'model': best_ml_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'model_name': 'Gradient Boosting',
        'roc_auc': ml_results['Gradient Boosting']['roc_auc']
    }
    
    with open('credit_default_ml_model.pkl', 'wb') as f:
        pickle.dump(ml_artifacts, f)
    print("\n✓ ML model saved as 'credit_default_ml_model.pkl'")
    
    # Save DNN model
    dnn_model.save('credit_default_dnn_model.h5')
    
    dnn_artifacts = {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'roc_auc': dnn_auc
    }
    
    with open('credit_default_dnn_artifacts.pkl', 'wb') as f:
        pickle.dump(dnn_artifacts, f)
    print("✓ DNN model saved as 'credit_default_dnn_model.h5'")
    print("✓ DNN artifacts saved as 'credit_default_dnn_artifacts.pkl'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Generate data
    df = generate_credit_data(10000)
    
    # EDA
    perform_eda(df)
    
    # Feature engineering
    df_fe = feature_engineering(df)
    
    # Preprocessing
    X_train, X_test, y_train, y_test, scaler, feature_cols, encoders = preprocess_data(df_fe)
    
    # Train ML models
    ml_results = train_ml_models(X_train, X_test, y_train, y_test)
    
    # Train DNN
    dnn_model, dnn_auc, dnn_proba = train_deep_learning_model(X_train, X_test, y_train, y_test)
    
    # Compare and save
    compare_and_save_models(ml_results, dnn_model, dnn_auc, dnn_proba, y_test,
                           scaler, feature_cols, encoders)
    
    print("\n" + "="*80)
    print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. credit_eda_analysis.png - EDA visualizations")
    print("  2. dnn_training_history.png - DNN training curves")
    print("  3. model_comparison_roc.png - ROC curve comparison")
    print("  4. credit_default_ml_model.pkl - Best ML model")
    print("  5. credit_default_dnn_model.h5 - Deep Neural Network")
    print("  6. credit_default_dnn_artifacts.pkl - DNN preprocessing artifacts")
    print("\nNext Step: Run the Streamlit app (app.py)")
    print("="*80)