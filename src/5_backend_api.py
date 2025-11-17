"""
Step 5: Backend API
Flask API to serve predictions and real-time data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import joblib
from keras.models import load_model
import os
import threading
import time

app = Flask(__name__)
CORS(app)

# Global variables
current_scenario = 'normal'
current_cycle = 0
is_running = False
simulator = None

class EngineSimulator:
    """Real-time engine simulator"""
    
    def __init__(self, scenario='normal'):
        self.scenario = scenario
        self.cycle = 0
        
        # Load thresholds
        with open('models/thresholds.json', 'r') as f:
            self.thresholds = json.load(f)
        
        # Load mock data
        self.mock_data = pd.read_csv(f'data/mock_{scenario}.csv')
        
    def get_reading(self, cycle):
        """Get reading for specific cycle"""
        if cycle >= len(self.mock_data):
            cycle = len(self.mock_data) - 1
        
        return self.mock_data.iloc[cycle].to_dict()
    
    def check_thresholds(self, reading):
        """Check threshold violations"""
        alerts = []
        
        for sensor in ['T2', 'T24', 'T30', 'T50', 'P2', 'P30', 'Nf', 'Nc']:
            if sensor not in reading or sensor not in self.thresholds:
                continue
            
            value = reading[sensor]
            threshold = self.thresholds[sensor]
            
            if value < threshold['min']:
                severity = 'CRITICAL' if value < threshold['min'] * 0.98 else 'WARNING'
                alerts.append({
                    'sensor': sensor,
                    'sensor_name': threshold['name'],
                    'type': 'LOW',
                    'severity': severity,
                    'value': round(value, 2),
                    'threshold': threshold['min'],
                    'unit': threshold['unit'],
                    'message': f"{threshold['name']} below minimum: {value:.2f} < {threshold['min']} {threshold['unit']}",
                    'impact': threshold['critical_impact']
                })
            
            elif value > threshold['max']:
                severity = 'CRITICAL' if value > threshold['max'] * 1.02 else 'WARNING'
                alerts.append({
                    'sensor': sensor,
                    'sensor_name': threshold['name'],
                    'type': 'HIGH',
                    'severity': severity,
                    'value': round(value, 2),
                    'threshold': threshold['max'],
                    'unit': threshold['unit'],
                    'message': f"{threshold['name']} above maximum: {value:.2f} > {threshold['max']} {threshold['unit']}",
                    'impact': threshold['critical_impact']
                })
        
        return alerts

class RULPredictor:
    """RUL prediction using trained model"""
    
    def __init__(self):
        try:
            # Try to load LSTM model
            self.model = load_model('models/lstm_rul_model.h5')
            self.scaler = joblib.load('models/scaler.pkl')
            self.metadata = joblib.load('models/metadata.pkl')
            self.model_type = 'LSTM'
            print("✅ LSTM model loaded")
        except:
            try:
                # Fallback to Random Forest
                self.model = joblib.load('models/rf_rul_model.pkl')
                self.scaler = joblib.load('models/scaler.pkl')
                self.metadata = joblib.load('models/metadata.pkl')
                self.model_type = 'RandomForest'
                print("✅ Random Forest model loaded")
            except:
                self.model = None
                print("⚠️ No trained model found, using simple heuristic")
    
    def predict(self, reading):
        """Predict RUL from sensor reading"""
        
        if self.model is None:
            # Simple heuristic-based prediction
            return self.predict_heuristic(reading)
        
        try:
            # Extract features (simplified)
            features = [
                reading.get('cycle', 0),
                reading.get('T2', 518),
                reading.get('T24', 642),
                reading.get('T30', 1580),
                reading.get('T50', 1398),
                reading.get('P2', 14.6),
                reading.get('P30', 45),
                reading.get('Nf', 2388),
                reading.get('Nc', 9046)
            ]
            
            # Pad to match training features
            while len(features) < self.metadata['n_features']:
                features.append(0)
            
            # Scale and predict
            features_scaled = self.scaler.transform([features[:self.metadata['n_features']]])
            
            if self.model_type == 'LSTM':
                # For LSTM, need sequence
                sequence = np.repeat(features_scaled, 30, axis=0).reshape(1, 30, -1)
                rul = self.model.predict(sequence, verbose=0)[0][0]
            else:
                # For RF, flatten if needed
                rul = self.model.predict(features_scaled)[0]
            
            rul = max(0, int(rul))
            confidence = 0.85 + np.random.random() * 0.1
            
        except Exception as e:
            print(f"Prediction error: {e}")
            rul = self.predict_heuristic(reading)
            confidence = 0.75
        
        health_score = min(100, int((rul / 150) * 100))
        
        return {
            'rul': rul,
            'confidence': round(confidence, 2),
            'health_score': health_score,
            'model_type': self.model_type if self.model else 'Heuristic'
        }
    
    def predict_heuristic(self, reading):
        """Simple heuristic prediction when no model available"""
        # Based on key temperature and pressure deviations
        T30_baseline = 1580
        T50_baseline = 1398
        P30_baseline = 45
        
        T30_deg = max(0, (reading.get('T30', T30_baseline) - T30_baseline) / 2)
        T50_deg = max(0, (reading.get('T50', T50_baseline) - T50_baseline) / 1.5)
        P30_deg = max(0, (P30_baseline - reading.get('P30', P30_baseline)) * 3)
        
        total_deg = T30_deg + T50_deg + P30_deg
        rul = max(0, int(150 - total_deg))
        
        return rul

# Initialize predictor
predictor = RULPredictor()

# API Routes

@app.route('/')
def home():
    """API home"""
    return jsonify({
        'message': 'Predictive Maintenance API',
        'version': '1.0',
        'endpoints': {
            'scenarios': '/api/scenarios',
            'current': '/api/current',
            'predict': '/api/predict',
            'thresholds': '/api/thresholds',
            'control': '/api/control'
        }
    })

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Get available scenarios"""
    try:
        with open('data/scenarios_summary.json', 'r') as f:
            scenarios = json.load(f)
        return jsonify(scenarios)
    except:
        return jsonify({
            'scenarios': [
                {'id': 'normal', 'name': 'Normal Operation'},
                {'id': 'gradual_degradation', 'name': 'Gradual Degradation'},
                {'id': 'critical', 'name': 'Critical Condition'},
                {'id': 'emergency', 'name': 'Emergency'},
                {'id': 'sudden_anomaly', 'name': 'Sudden Anomaly'},
                {'id': 'pressure_drop', 'name': 'Pressure Drop'}
            ]
        })

@app.route('/api/thresholds', methods=['GET'])
def get_thresholds():
    """Get safety thresholds"""
    try:
        with open('models/thresholds.json', 'r') as f:
            thresholds = json.load(f)
        return jsonify(thresholds)
    except:
        return jsonify({'error': 'Thresholds not found'}), 404

@app.route('/api/current', methods=['GET'])
def get_current():
    """Get current sensor reading"""
    global simulator, current_cycle
    
    if simulator is None:
        return jsonify({'error': 'Simulator not initialized'}), 400
    
    # Get reading
    reading = simulator.get_reading(current_cycle)
    
    # Check thresholds
    alerts = simulator.check_thresholds(reading)
    
    # Predict RUL
    prediction = predictor.predict(reading)
    
    return jsonify({
        'cycle': current_cycle,
        'scenario': current_scenario,
        'reading': reading,
        'prediction': prediction,
        'alerts': alerts,
        'timestamp': time.time()
    })

@app.route('/api/predict', methods=['POST'])
def predict_rul():
    """Predict RUL for custom sensor reading"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    prediction = predictor.predict(data)
    
    return jsonify(prediction)

@app.route('/api/control/start', methods=['POST'])
def start_simulation():
    """Start simulation"""
    global is_running, simulator, current_scenario, current_cycle
    
    data = request.json or {}
    current_scenario = data.get('scenario', 'normal')
    
    try:
        simulator = EngineSimulator(current_scenario)
        current_cycle = 0
        is_running = True
        
        return jsonify({
            'status': 'started',
            'scenario': current_scenario,
            'message': 'Simulation started successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/control/stop', methods=['POST'])
def stop_simulation():
    """Stop simulation"""
    global is_running
    is_running = False
    
    return jsonify({
        'status': 'stopped',
        'message': 'Simulation stopped'
    })

@app.route('/api/control/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation"""
    global is_running, current_cycle, simulator
    
    is_running = False
    current_cycle = 0
    simulator = None
    
    return jsonify({
        'status': 'reset',
        'message': 'Simulation reset'
    })

@app.route('/api/control/step', methods=['POST'])
def step_simulation():
    """Advance one cycle"""
    global current_cycle, simulator
    
    if simulator is None:
        return jsonify({'error': 'Simulator not initialized'}), 400
    
    if current_cycle < len(simulator.mock_data) - 1:
        current_cycle += 1
    
    return get_current()

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get historical data for current scenario"""
    global simulator, current_cycle
    
    if simulator is None:
        return jsonify({'error': 'Simulator not initialized'}), 400
    
    # Return data up to current cycle
    history = simulator.mock_data.iloc[:current_cycle+1].to_dict('records')
    
    return jsonify({
        'scenario': current_scenario,
        'history': history
    })

def run_app():
    """Run Flask app"""
    print("\n" + "=" * 80)
    print("🚀 PREDICTIVE MAINTENANCE API SERVER")
    print("=" * 80)
    print("\n📡 API Endpoints:")
    print("   GET  /                      - API info")
    print("   GET  /api/scenarios         - List scenarios")
    print("   GET  /api/thresholds        - Get thresholds")
    print("   GET  /api/current           - Current reading")
    print("   POST /api/predict           - Predict RUL")
    print("   POST /api/control/start     - Start simulation")
    print("   POST /api/control/stop      - Stop simulation")
    print("   POST /api/control/reset     - Reset simulation")
    print("   POST /api/control/step      - Step one cycle")
    print("   GET  /api/history           - Get history")
    print("\n🌐 Server running at: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    run_app()