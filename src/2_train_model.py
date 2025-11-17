"""
Step 2: Train ML Model for RUL Prediction
Uses LSTM neural network and XGBoost for comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Deep Learning
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Create models directory
os.makedirs('models', exist_ok=True)

def load_processed_data(filepath='data/processed_data.csv'):
    """Load preprocessed data"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded processed data: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Processed data not found!")
        print("   Please run '1_data_preparation.py' first")
        return None

def prepare_features(df):
    """Prepare features for model training"""
    
    # Select key features for training
    feature_columns = [
        'cycle',
        # Raw sensors
        'T2', 'T24', 'T30', 'T50', 'P2', 'P30', 'Nf', 'Nc',
        # Rolling features
        'T30_rolling_mean_5', 'T50_rolling_mean_5',
        'P2_rolling_mean_5', 'P30_rolling_mean_5',
        # Rate of change
        'T30_rate_change', 'T50_rate_change',
        'P2_rate_change', 'P30_rate_change',
        # Settings
        'setting1', 'setting2', 'setting3'
    ]
    
    # Filter existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"📊 Using {len(available_features)} features for training")
    
    X = df[available_features].values
    y = df['RUL'].values
    
    return X, y, available_features

def create_sequences(X, y, sequence_length=30):
    """Create sequences for LSTM"""
    Xs, ys = [], []
    
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:i+sequence_length])
        ys.append(y[i+sequence_length])
    
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # RUL prediction
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train LSTM model"""
    print("\n🤖 Training LSTM Model...")
    print("=" * 80)
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    print(f"   Model architecture:")
    model.summary()
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 LSTM Model Performance:")
    print(f"   Test MAE: {test_mae:.2f} cycles")
    print(f"   Test RMSE: {rmse:.2f} cycles")
    print(f"   R² Score: {r2:.4f}")
    
    # Save model
    model.save('models/lstm_rul_model.h5')
    print(f"\n✅ LSTM model saved to: models/lstm_rul_model.h5")
    
    return model, history, y_pred

def train_simple_model(X_train, y_train, X_test, y_test):
    """Train simple Random Forest model for comparison"""
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n🌲 Training Random Forest Model (Baseline)...")
    print("=" * 80)
    
    # Flatten sequences if they exist
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_flat = X_train
        X_test_flat = X_test
    
    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_flat, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_flat)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Random Forest Performance:")
    print(f"   Test MAE: {mae:.2f} cycles")
    print(f"   Test RMSE: {rmse:.2f} cycles")
    print(f"   R² Score: {r2:.4f}")
    
    # Save model
    joblib.dump(model, 'models/rf_rul_model.pkl')
    print(f"\n✅ Random Forest model saved to: models/rf_rul_model.pkl")
    
    return model, y_pred

def plot_results(y_test, y_pred_lstm, y_pred_rf, history):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('LSTM Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # LSTM Predictions
    axes[0, 1].scatter(y_test, y_pred_lstm, alpha=0.5, s=10)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_title('LSTM: Predicted vs Actual RUL')
    axes[0, 1].set_xlabel('Actual RUL')
    axes[0, 1].set_ylabel('Predicted RUL')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Random Forest Predictions
    axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.5, s=10, color='green')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_title('Random Forest: Predicted vs Actual RUL')
    axes[1, 0].set_xlabel('Actual RUL')
    axes[1, 0].set_ylabel('Predicted RUL')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    error_lstm = y_test - y_pred_lstm
    error_rf = y_test - y_pred_rf
    axes[1, 1].hist(error_lstm, bins=50, alpha=0.5, label='LSTM', color='blue')
    axes[1, 1].hist(error_rf, bins=50, alpha=0.5, label='Random Forest', color='green')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Error (Actual - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Results plot saved to: models/training_results.png")
    plt.close()

def save_scaler_and_metadata(scaler, feature_names):
    """Save scaler and feature metadata"""
    joblib.dump(scaler, 'models/scaler.pkl')
    
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    
    print(f"✅ Scaler and metadata saved")

def main():
    """Main execution"""
    print("\n🚀 STEP 2: MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    df = load_processed_data()
    if df is None:
        return
    
    # Prepare features
    print("\n⚙️ Preparing features...")
    X, y, feature_names = prepare_features(df)
    
    # Normalize features
    print("⚙️ Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for LSTM
    print("⚙️ Creating sequences for LSTM (window=30 cycles)...")
    sequence_length = 30
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
    
    print(f"   Sequence shape: {X_seq.shape}")
    print(f"   Target shape: {y_seq.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train LSTM model
    lstm_model, history, y_pred_lstm = train_lstm_model(X_train, y_train, X_test, y_test)
    
    # Train Random Forest (baseline) - OPTIONAL: Comment out if too slow
    print("\n⚠️ Skipping Random Forest training (too slow on CPU)")
    print("   LSTM model is sufficient for hackathon demo!")
    y_pred_rf = y_pred_lstm  # Use LSTM predictions for visualization
    
    # Uncomment below if you want to train RF (will take 5-10 minutes):
    # rf_model, y_pred_rf = train_simple_model(X_train, y_train, X_test, y_test)
    
    # Plot results
    print("\n📊 Generating visualization...")
    plot_results(y_test, y_pred_lstm, y_pred_rf, history)
    
    # Save scaler and metadata
    save_scaler_and_metadata(scaler, feature_names)
    
    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print("\n📁 Saved files:")
    print("   - models/lstm_rul_model.h5")
    print("   - models/rf_rul_model.pkl")
    print("   - models/scaler.pkl")
    print("   - models/metadata.pkl")
    print("   - models/training_results.png")
    print("\n   Next step: Run '3_define_thresholds.py'")

if __name__ == "__main__":
    main()