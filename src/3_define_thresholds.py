"""
Step 3: Define Safety Thresholds
Analyze historical data to set warning and critical thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

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

def analyze_sensor_ranges(df):
    """Analyze sensor value ranges across healthy and failing engines"""
    
    print("\n🔍 ANALYZING SENSOR RANGES")
    print("=" * 80)
    
    key_sensors = ['T2', 'T24', 'T30', 'T50', 'P2', 'P30', 'Nf', 'Nc']
    
    # Separate healthy (RUL > 100) vs degraded (RUL < 30) data
    healthy = df[df['RUL'] > 100]
    degraded = df[df['RUL'] < 30]
    
    sensor_stats = {}
    
    print(f"\n📊 Comparing Healthy (RUL>100) vs Degraded (RUL<30) Engines:")
    print(f"   Healthy samples: {len(healthy)}")
    print(f"   Degraded samples: {len(degraded)}")
    print("\n" + "-" * 80)
    
    for sensor in key_sensors:
        if sensor not in df.columns:
            continue
            
        # Statistics for healthy engines
        healthy_mean = healthy[sensor].mean()
        healthy_std = healthy[sensor].std()
        healthy_min = healthy[sensor].min()
        healthy_max = healthy[sensor].max()
        
        # Statistics for degraded engines
        degraded_mean = degraded[sensor].mean()
        degraded_std = degraded[sensor].std()
        degraded_min = degraded[sensor].min()
        degraded_max = degraded[sensor].max()
        
        # Overall statistics
        overall_mean = df[sensor].mean()
        overall_std = df[sensor].std()
        percentile_5 = df[sensor].quantile(0.05)
        percentile_95 = df[sensor].quantile(0.95)
        
        sensor_stats[sensor] = {
            'healthy_mean': healthy_mean,
            'healthy_std': healthy_std,
            'healthy_range': (healthy_min, healthy_max),
            'degraded_mean': degraded_mean,
            'degraded_std': degraded_std,
            'degraded_range': (degraded_min, degraded_max),
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95
        }
        
        print(f"\n{sensor}:")
        print(f"   Healthy:  {healthy_mean:.2f} ± {healthy_std:.2f} (range: {healthy_min:.2f} - {healthy_max:.2f})")
        print(f"   Degraded: {degraded_mean:.2f} ± {degraded_std:.2f} (range: {degraded_min:.2f} - {degraded_max:.2f})")
        print(f"   Change:   {((degraded_mean - healthy_mean) / healthy_mean * 100):+.2f}%")
    
    return sensor_stats

def define_thresholds(sensor_stats):
    """Define warning and critical thresholds based on statistics"""
    
    print("\n\n🎯 DEFINING SAFETY THRESHOLDS")
    print("=" * 80)
    print("\nThreshold Logic:")
    print("   WARNING:  ±2 standard deviations from healthy mean")
    print("   CRITICAL: ±3 standard deviations from healthy mean")
    print("   EMERGENCY: Values seen in degraded engines (RUL < 30)")
    print("\n" + "-" * 80)
    
    thresholds = {}
    
    # Sensor units
    units = {
        'T2': '°R', 'T24': '°R', 'T30': '°R', 'T50': '°R',
        'P2': 'psia', 'P30': 'psia',
        'Nf': 'rpm', 'Nc': 'rpm'
    }
    
    for sensor, stats in sensor_stats.items():
        # Calculate thresholds
        healthy_mean = stats['healthy_mean']
        healthy_std = stats['healthy_std']
        
        # For temperature sensors: increase indicates degradation
        # For pressure sensors: decrease indicates degradation
        if sensor.startswith('T'):  # Temperature
            warning_low = healthy_mean - 2 * healthy_std
            warning_high = healthy_mean + 2 * healthy_std
            critical_low = healthy_mean - 3 * healthy_std
            critical_high = healthy_mean + 3 * healthy_std
            emergency_high = stats['degraded_range'][1]
        else:  # Pressure and Speed
            warning_low = healthy_mean - 2 * healthy_std
            warning_high = healthy_mean + 2 * healthy_std
            critical_low = healthy_mean - 3 * healthy_std
            critical_high = healthy_mean + 3 * healthy_std
            emergency_high = warning_high * 1.05
        
        thresholds[sensor] = {
            'normal_range': {
                'min': round(warning_low, 2),
                'max': round(warning_high, 2)
            },
            'warning_range': {
                'min': round(critical_low, 2),
                'max': round(critical_high, 2)
            },
            'critical_max': round(emergency_high, 2),
            'unit': units.get(sensor, ''),
            'healthy_baseline': round(healthy_mean, 2),
            'degradation_direction': 'increase' if sensor.startswith('T') else 'decrease/increase'
        }
        
        print(f"\n{sensor} ({units.get(sensor, '')}):")
        print(f"   Baseline (healthy): {healthy_mean:.2f}")
        print(f"   Normal range:       {thresholds[sensor]['normal_range']['min']:.2f} - {thresholds[sensor]['normal_range']['max']:.2f}")
        print(f"   Warning threshold:  {thresholds[sensor]['warning_range']['min']:.2f} - {thresholds[sensor]['warning_range']['max']:.2f}")
        print(f"   Critical max:       {thresholds[sensor]['critical_max']:.2f}")
    
    return thresholds

def create_simplified_thresholds():
    """Create simplified, practical thresholds for demo"""
    
    print("\n\n📋 CREATING SIMPLIFIED THRESHOLDS FOR DEMO")
    print("=" * 80)
    print("These are practical ranges based on NASA documentation and expert knowledge")
    print("-" * 80)
    
    thresholds = {
        'T2': {
            'min': 515.0,
            'max': 525.0,
            'unit': '°R',
            'name': 'Fan Inlet Temperature',
            'critical_impact': 'Intake air temperature anomaly'
        },
        'T24': {
            'min': 635.0,
            'max': 650.0,
            'unit': '°R',
            'name': 'LPC Outlet Temperature',
            'critical_impact': 'Low pressure compressor efficiency'
        },
        'T30': {
            'min': 1570.0,
            'max': 1600.0,
            'unit': '°R',
            'name': 'HPC Outlet Temperature',
            'critical_impact': 'High pressure compressor health - CRITICAL PARAMETER'
        },
        'T50': {
            'min': 1385.0,
            'max': 1420.0,
            'unit': '°R',
            'name': 'LPT Outlet Temperature',
            'critical_impact': 'Turbine blade degradation - CRITICAL PARAMETER'
        },
        'P2': {
            'min': 14.0,
            'max': 15.5,
            'unit': 'psia',
            'name': 'Fan Inlet Pressure',
            'critical_impact': 'Inlet pressure anomaly'
        },
        'P30': {
            'min': 43.0,
            'max': 47.0,
            'unit': 'psia',
            'name': 'HPC Outlet Pressure',
            'critical_impact': 'Compressor pressure ratio - indicates seal wear'
        },
        'Nf': {
            'min': 2300.0,
            'max': 2500.0,
            'unit': 'rpm',
            'name': 'Fan Speed',
            'critical_impact': 'Fan imbalance or bearing issues'
        },
        'Nc': {
            'min': 9000.0,
            'max': 9500.0,
            'unit': 'rpm',
            'name': 'Core Speed',
            'critical_impact': 'Core rotor imbalance'
        }
    }
    
    for sensor, info in thresholds.items():
        print(f"\n{sensor} - {info['name']}")
        print(f"   Range: {info['min']} - {info['max']} {info['unit']}")
        print(f"   Impact: {info['critical_impact']}")
    
    return thresholds

def save_thresholds(thresholds, filepath='models/thresholds.json'):
    """Save thresholds to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\n✅ Thresholds saved to: {filepath}")

def plot_threshold_visualization(df, thresholds):
    """Visualize thresholds on actual data"""
    
    key_sensors = ['T30', 'T50', 'P2', 'P30']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, sensor in enumerate(key_sensors):
        if sensor not in df.columns:
            continue
        
        ax = axes[idx]
        thresh = thresholds[sensor]
        
        # Plot data by RUL category
        healthy = df[df['RUL'] > 100]
        warning = df[(df['RUL'] <= 100) & (df['RUL'] > 30)]
        critical = df[df['RUL'] <= 30]
        
        ax.scatter(healthy['cycle'], healthy[sensor], s=1, alpha=0.3, c='green', label='Healthy (RUL>100)')
        ax.scatter(warning['cycle'], warning[sensor], s=1, alpha=0.5, c='orange', label='Warning (30<RUL≤100)')
        ax.scatter(critical['cycle'], critical[sensor], s=1, alpha=0.7, c='red', label='Critical (RUL≤30)')
        
        # Plot thresholds
        ax.axhline(y=thresh['min'], color='yellow', linestyle='--', linewidth=2, label='Warning Min')
        ax.axhline(y=thresh['max'], color='yellow', linestyle='--', linewidth=2, label='Warning Max')
        
        ax.set_title(f"{sensor} - {thresh['name']}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Cycle')
        ax.set_ylabel(f'{sensor} ({thresh["unit"]})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/threshold_visualization.png', dpi=300, bbox_inches='tight')
    print(f"📊 Threshold visualization saved to: models/threshold_visualization.png")
    plt.close()

def main():
    """Main execution"""
    print("\n🚀 STEP 3: DEFINE SAFETY THRESHOLDS")
    print("=" * 80)
    
    # Load data
    df = load_processed_data()
    if df is None:
        return
    
    # Analyze sensor ranges
    sensor_stats = analyze_sensor_ranges(df)
    
    # Define statistical thresholds
    statistical_thresholds = define_thresholds(sensor_stats)
    
    # Create simplified practical thresholds
    practical_thresholds = create_simplified_thresholds()
    
    # Save both versions
    save_thresholds(statistical_thresholds, 'models/thresholds_statistical.json')
    save_thresholds(practical_thresholds, 'models/thresholds.json')
    
    # Visualize
    print("\n📊 Creating threshold visualization...")
    plot_threshold_visualization(df, practical_thresholds)
    
    print("\n" + "=" * 80)
    print("✅ THRESHOLD DEFINITION COMPLETE!")
    print("=" * 80)
    print("\n📁 Saved files:")
    print("   - models/thresholds.json (practical thresholds for demo)")
    print("   - models/thresholds_statistical.json (data-driven thresholds)")
    print("   - models/threshold_visualization.png")
    print("\n   Next step: Run '4_mock_data_generator.py'")

if __name__ == "__main__":
    main()