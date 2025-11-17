"""
Step 4: Mock Data Generator
Create realistic test scenarios for real-time simulation
"""

import pandas as pd
import numpy as np
import json
import os

class EngineSimulator:
    """Simulate turbofan engine sensor data"""
    
    def __init__(self, scenario='normal'):
        self.scenario = scenario
        self.cycle = 0
        
        # Load thresholds
        with open('models/thresholds.json', 'r') as f:
            self.thresholds = json.load(f)
        
        # Define scenario parameters
        self.scenarios = {
            'normal': {
                'name': 'Normal Operation',
                'description': 'All parameters within safe ranges',
                'base_values': {
                    'T2': 518.67,
                    'T24': 642.15,
                    'T30': 1580.5,
                    'T50': 1398.2,
                    'P2': 14.62,
                    'P30': 45.2,
                    'Nf': 2388.06,
                    'Nc': 9046.19
                },
                'variance': 0.005,  # ±0.5% random variation
                'trend': 0.0,        # No degradation trend
                'expected_rul': 150
            },
            'gradual_degradation': {
                'name': 'Gradual Degradation',
                'description': 'Temperature rising, pressure declining slowly',
                'base_values': {
                    'T2': 520.2,
                    'T24': 645.8,
                    'T30': 1585.3,
                    'T50': 1405.5,
                    'P2': 14.45,
                    'P30': 44.5,
                    'Nf': 2395.3,
                    'Nc': 9062.8
                },
                'variance': 0.01,
                'trend': 0.001,      # 0.1% increase per cycle
                'expected_rul': 80
            },
            'critical': {
                'name': 'Critical Condition',
                'description': 'Multiple parameters approaching warning levels',
                'base_values': {
                    'T2': 522.5,
                    'T24': 648.9,
                    'T30': 1595.2,
                    'T50': 1415.3,
                    'P2': 14.25,
                    'P30': 43.8,
                    'Nf': 2405.1,
                    'Nc': 9088.5
                },
                'variance': 0.015,
                'trend': 0.002,
                'expected_rul': 30
            },
            'emergency': {
                'name': 'Emergency - Imminent Failure',
                'description': 'Threshold breaches, immediate maintenance required',
                'base_values': {
                    'T2': 524.8,
                    'T24': 650.5,
                    'T30': 1602.7,
                    'T50': 1422.2,
                    'P2': 14.05,
                    'P30': 43.2,
                    'Nf': 2415.6,
                    'Nc': 9105.3
                },
                'variance': 0.02,
                'trend': 0.003,
                'expected_rul': 10
            },
            'sudden_anomaly': {
                'name': 'Sudden Anomaly',
                'description': 'Sudden spike in temperature (bird strike or FOD)',
                'base_values': {
                    'T2': 518.67,
                    'T24': 642.15,
                    'T30': 1580.5,
                    'T50': 1398.2,
                    'P2': 14.62,
                    'P30': 45.2,
                    'Nf': 2388.06,
                    'Nc': 9046.19
                },
                'variance': 0.01,
                'trend': 0.0,
                'anomaly_cycle': 50,  # Anomaly occurs at cycle 50
                'anomaly_sensors': ['T30', 'T50'],
                'anomaly_magnitude': 0.03,  # 3% sudden increase
                'expected_rul': 100
            },
            'pressure_drop': {
                'name': 'Pressure System Failure',
                'description': 'Progressive pressure loss (seal failure)',
                'base_values': {
                    'T2': 519.5,
                    'T24': 643.8,
                    'T30': 1582.0,
                    'T50': 1400.1,
                    'P2': 14.55,
                    'P30': 45.0,
                    'Nf': 2390.0,
                    'Nc': 9050.0
                },
                'variance': 0.008,
                'trend': -0.002,     # Pressure decreasing
                'trend_sensors': ['P2', 'P30'],
                'expected_rul': 50
            }
        }
    
    def generate_reading(self, cycle_num=None):
        """Generate a single sensor reading"""
        if cycle_num is not None:
            self.cycle = cycle_num
        else:
            self.cycle += 1
        
        config = self.scenarios[self.scenario]
        reading = {'cycle': self.cycle}
        
        for sensor, base_value in config['base_values'].items():
            # Base value with random variance
            variance = config['variance']
            random_factor = 1 + (np.random.random() - 0.5) * 2 * variance
            value = base_value * random_factor
            
            # Apply degradation trend
            trend = config.get('trend', 0)
            
            # Special handling for different scenarios
            if self.scenario == 'sudden_anomaly':
                # Sudden spike at specific cycle
                if self.cycle >= config.get('anomaly_cycle', 50):
                    if sensor in config.get('anomaly_sensors', []):
                        spike = base_value * config.get('anomaly_magnitude', 0.03)
                        value += spike
            
            elif self.scenario == 'pressure_drop':
                # Pressure-specific degradation
                if sensor in config.get('trend_sensors', []):
                    trend_factor = 1 + (trend * self.cycle)
                    value *= trend_factor
                else:
                    # Temperature sensors increase slightly
                    if sensor.startswith('T'):
                        trend_factor = 1 + (abs(trend) * 0.5 * self.cycle)
                        value *= trend_factor
            
            else:
                # Normal trend application
                if sensor.startswith('T'):  # Temperature increases
                    trend_factor = 1 + (trend * self.cycle)
                else:  # Pressure/speed: decrease for degradation
                    trend_factor = 1 - (trend * 0.5 * self.cycle)
                
                value *= trend_factor
            
            reading[sensor] = round(value, 2)
        
        # Add metadata
        reading['scenario'] = self.scenario
        reading['scenario_name'] = config['name']
        
        return reading
    
    def generate_sequence(self, num_cycles=100):
        """Generate a sequence of readings"""
        readings = []
        for i in range(1, num_cycles + 1):
            reading = self.generate_reading(i)
            readings.append(reading)
        
        return pd.DataFrame(readings)
    
    def check_thresholds(self, reading):
        """Check if reading violates any thresholds"""
        alerts = []
        
        for sensor in ['T2', 'T24', 'T30', 'T50', 'P2', 'P30', 'Nf', 'Nc']:
            if sensor not in reading or sensor not in self.thresholds:
                continue
            
            value = reading[sensor]
            threshold = self.thresholds[sensor]
            
            # Check violations
            if value < threshold['min']:
                severity = 'CRITICAL' if value < threshold['min'] * 0.98 else 'WARNING'
                alerts.append({
                    'sensor': sensor,
                    'type': 'LOW',
                    'severity': severity,
                    'value': value,
                    'threshold': threshold['min'],
                    'message': f"{sensor} below minimum: {value:.2f} < {threshold['min']} {threshold['unit']}"
                })
            
            elif value > threshold['max']:
                severity = 'CRITICAL' if value > threshold['max'] * 1.02 else 'WARNING'
                alerts.append({
                    'sensor': sensor,
                    'type': 'HIGH',
                    'severity': severity,
                    'value': value,
                    'threshold': threshold['max'],
                    'message': f"{sensor} above maximum: {value:.2f} > {threshold['max']} {threshold['unit']}"
                })
        
        return alerts

def generate_all_scenarios():
    """Generate data for all scenarios"""
    
    print("\n🎲 GENERATING MOCK DATA FOR ALL SCENARIOS")
    print("=" * 80)
    
    scenarios = ['normal', 'gradual_degradation', 'critical', 'emergency', 
                 'sudden_anomaly', 'pressure_drop']
    
    all_data = {}
    
    for scenario in scenarios:
        print(f"\n📊 Generating: {scenario}")
        simulator = EngineSimulator(scenario)
        
        # Generate 100 cycles
        df = simulator.generate_sequence(num_cycles=100)
        
        # Add threshold violations
        alerts_summary = {'WARNING': 0, 'CRITICAL': 0}
        for idx, row in df.iterrows():
            alerts = simulator.check_thresholds(row.to_dict())
            for alert in alerts:
                alerts_summary[alert['severity']] += 1
        
        print(f"   Cycles generated: {len(df)}")
        print(f"   Warnings: {alerts_summary['WARNING']}")
        print(f"   Critical alerts: {alerts_summary['CRITICAL']}")
        
        # Save to CSV
        filepath = f'data/mock_{scenario}.csv'
        df.to_csv(filepath, index=False)
        print(f"   Saved to: {filepath}")
        
        all_data[scenario] = df
    
    return all_data

def create_scenario_summary():
    """Create a summary document of all scenarios"""
    
    simulator = EngineSimulator('normal')
    scenarios = simulator.scenarios
    
    summary = {
        'scenarios': []
    }
    
    for scenario_key, config in scenarios.items():
        scenario_info = {
            'id': scenario_key,
            'name': config['name'],
            'description': config['description'],
            'expected_rul': config.get('expected_rul', 100),
            'characteristics': {
                'variance': f"±{config['variance']*100:.1f}%",
                'trend': f"{config.get('trend', 0)*100:.2f}% per cycle"
            },
            'typical_alerts': []
        }
        
        # Determine typical alerts
        if scenario_key == 'normal':
            scenario_info['typical_alerts'] = ['None - all parameters within limits']
        elif scenario_key == 'gradual_degradation':
            scenario_info['typical_alerts'] = ['T30 trending upward', 'P30 decreasing slightly']
        elif scenario_key == 'critical':
            scenario_info['typical_alerts'] = ['T30 near threshold', 'T50 elevated', 'Multiple warnings']
        elif scenario_key == 'emergency':
            scenario_info['typical_alerts'] = ['T30 exceeds threshold', 'Critical alerts', 'Immediate action required']
        elif scenario_key == 'sudden_anomaly':
            scenario_info['typical_alerts'] = ['Sudden T30 spike at cycle 50', 'FOD event simulation']
        elif scenario_key == 'pressure_drop':
            scenario_info['typical_alerts'] = ['Progressive pressure loss', 'Seal failure indicator']
        
        summary['scenarios'].append(scenario_info)
    
    # Save summary
    with open('data/scenarios_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Scenario summary saved to: data/scenarios_summary.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SCENARIO SUMMARY")
    print("=" * 80)
    
    for scenario in summary['scenarios']:
        print(f"\n{scenario['name']} ({scenario['id']})")
        print(f"   Description: {scenario['description']}")
        print(f"   Expected RUL: {scenario['expected_rul']} cycles")
        print(f"   Typical alerts: {', '.join(scenario['typical_alerts'])}")

def main():
    """Main execution"""
    print("\n🚀 STEP 4: MOCK DATA GENERATION")
    print("=" * 80)
    
    # Check if thresholds exist
    if not os.path.exists('models/thresholds.json'):
        print("❌ Thresholds not found!")
        print("   Please run '3_define_thresholds.py' first")
        return
    
    # Generate all scenarios
    all_data = generate_all_scenarios()
    
    # Create summary
    create_scenario_summary()
    
    print("\n" + "=" * 80)
    print("✅ MOCK DATA GENERATION COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated files:")
    print("   - data/mock_normal.csv")
    print("   - data/mock_gradual_degradation.csv")
    print("   - data/mock_critical.csv")
    print("   - data/mock_emergency.csv")
    print("   - data/mock_sudden_anomaly.csv")
    print("   - data/mock_pressure_drop.csv")
    print("   - data/scenarios_summary.json")
    print("\n   Next step: Run '5_backend_api.py'")

if __name__ == "__main__":
    main()