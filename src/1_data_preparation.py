"""
Step 1: Data Preparation
Download NASA C-MAPSS dataset and prepare it for training
"""

import pandas as pd
import numpy as np
import os

# Column names for the dataset
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
               [f'sensor{i}' for i in range(1, 22)]

# Sensor mapping (based on NASA documentation)
sensor_mapping = {
    'sensor1': 'T2',      # Total temperature at fan inlet (°R)
    'sensor2': 'T24',     # Total temperature at LPC outlet (°R)
    'sensor3': 'T30',     # Total temperature at HPC outlet (°R)
    'sensor4': 'T50',     # Total temperature at LPT outlet (°R)
    'sensor5': 'P2',      # Pressure at fan inlet (psia)
    'sensor6': 'P15',     # Total pressure in bypass-duct (psia)
    'sensor7': 'P30',     # Total pressure at HPC outlet (psia)
    'sensor8': 'Nf',      # Physical fan speed (rpm)
    'sensor9': 'Nc',      # Physical core speed (rpm)
    'sensor10': 'epr',    # Engine pressure ratio
    'sensor11': 'Ps30',   # Static pressure at HPC outlet (psia)
    'sensor12': 'phi',    # Ratio of fuel flow to Ps30
    'sensor13': 'NRf',    # Corrected fan speed (rpm)
    'sensor14': 'NRc',    # Corrected core speed (rpm)
    'sensor15': 'BPR',    # Bypass ratio
    'sensor16': 'farB',   # Burner fuel-air ratio
    'sensor17': 'htBleed',# Bleed enthalpy
    'sensor18': 'Nf_dmd', # Demanded fan speed (rpm)
    'sensor19': 'PCNfR_dmd', # Demanded corrected fan speed (rpm)
    'sensor20': 'W31',    # HPT coolant bleed (lbm/s)
    'sensor21': 'W32'     # LPT coolant bleed (lbm/s)
}

def download_instructions():
    """Print instructions to download the dataset"""
    print("=" * 80)
    print("NASA C-MAPSS TURBOFAN ENGINE DATASET")
    print("=" * 80)
    print("\n📥 DOWNLOAD INSTRUCTIONS:\n")
    print("1. Visit: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    print("   OR: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print("\n2. Download 'train_FD001.txt' file")
    print("\n3. Place it in the 'data/' folder of this project")
    print("\n4. Run this script again after downloading")
    print("\n" + "=" * 80)

def load_data(filepath='data/train_FD001.txt'):
    """Load the NASA dataset"""
    try:
        # Read the space-separated file
        df = pd.read_csv(filepath, sep='\s+', header=None, names=column_names)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Engines: {df['engine_id'].nunique()}")
        print(f"   Total cycles: {len(df)}")
        return df
    except FileNotFoundError:
        print("❌ Dataset file not found!")
        download_instructions()
        return None

def calculate_rul(df):
    """Calculate Remaining Useful Life (RUL) for each cycle"""
    # Group by engine and calculate max cycles
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge with original dataframe
    df = df.merge(max_cycles, on='engine_id', how='left')
    
    # Calculate RUL (max_cycle - current_cycle)
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    return df

def add_features(df):
    """Add engineered features"""
    # Rename sensors to meaningful names
    for old_name, new_name in sensor_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    # Sort by engine_id and cycle for rolling calculations
    df = df.sort_values(['engine_id', 'cycle'])
    
    # Rolling averages (window of 5 cycles)
    sensors_to_process = ['T2', 'T24', 'T30', 'T50', 'P2', 'P30', 'Nf', 'Nc']
    
    for sensor in sensors_to_process:
        if sensor in df.columns:
            # Rolling mean
            df[f'{sensor}_rolling_mean_5'] = df.groupby('engine_id')[sensor].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{sensor}_rolling_std_5'] = df.groupby('engine_id')[sensor].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )
            
            # Lag features (previous cycle)
            df[f'{sensor}_lag_1'] = df.groupby('engine_id')[sensor].shift(1)
            
            # Rate of change
            df[f'{sensor}_rate_change'] = df.groupby('engine_id')[sensor].pct_change()
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df

def save_processed_data(df, filepath='data/processed_data.csv'):
    """Save processed data"""
    df.to_csv(filepath, index=False)
    print(f"\n✅ Processed data saved to: {filepath}")
    print(f"   Total features: {len(df.columns)}")

def analyze_data(df):
    """Perform basic analysis"""
    print("\n" + "=" * 80)
    print("DATA ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total engines: {df['engine_id'].nunique()}")
    print(f"   Total cycles: {len(df)}")
    print(f"   Average cycles per engine: {df.groupby('engine_id')['cycle'].max().mean():.1f}")
    print(f"   Min cycles: {df.groupby('engine_id')['cycle'].max().min()}")
    print(f"   Max cycles: {df.groupby('engine_id')['cycle'].max().max()}")
    
    # Sensor statistics
    print(f"\n🌡️ Key Sensor Ranges:")
    key_sensors = ['T2', 'T24', 'T30', 'T50', 'P2', 'P30', 'Nf', 'Nc']
    for sensor in key_sensors:
        if sensor in df.columns:
            print(f"   {sensor:6s}: {df[sensor].min():8.2f} - {df[sensor].max():8.2f} (mean: {df[sensor].mean():8.2f})")
    
    # RUL distribution
    print(f"\n⏰ RUL Distribution:")
    print(f"   Mean RUL: {df['RUL'].mean():.1f} cycles")
    print(f"   Max RUL: {df['RUL'].max()} cycles")
    print(f"   Engines with RUL < 50: {(df.groupby('engine_id')['RUL'].min() < 50).sum()}")

def main():
    """Main execution"""
    print("\n🚀 STEP 1: DATA PREPARATION")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    if df is None:
        return
    
    # Calculate RUL
    print("\n⚙️ Calculating Remaining Useful Life (RUL)...")
    df = calculate_rul(df)
    
    # Add features
    print("⚙️ Engineering features...")
    df = add_features(df)
    
    # Analyze
    analyze_data(df)
    
    # Save
    save_processed_data(df)
    
    print("\n✅ Data preparation complete!")
    print("   Next step: Run '2_train_model.py'")

if __name__ == "__main__":
    main()