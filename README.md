# SmartGardenAI 🌱

An intelligent IoT-based smart garden system that combines sensor data, machine learning, and automation to optimize plant growth and garden management.

## Features

- **🌡️ Environmental Monitoring**: Real-time temperature, humidity, soil moisture, and light intensity tracking
- **🤖 AI-Powered Insights**: Machine learning models for plant health prediction and growth optimization
- **💧 Automated Irrigation**: Smart watering system with weather-based scheduling
- **📊 Data Analytics**: Comprehensive dashboard with historical data and trends
- **🔔 Smart Notifications**: Alerts for maintenance, watering schedules, and plant health issues
- **🌦️ Weather Integration**: Real-time weather data integration for optimal care decisions
- **📱 Mobile App**: Cross-platform mobile application for remote monitoring
- **🔧 IoT Device Management**: Centralized control of sensors, actuators, and smart devices

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Dashboard │    │   AI Engine     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Backend API    │
                    │  (FastAPI)      │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  IoT Gateway    │    │  Database       │    │  ML Pipeline    │
│  (Raspberry Pi) │    │  (PostgreSQL)   │    │  (TensorFlow)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         └───────────────────────┐
                                 │
                    ┌─────────────────┐
                    │  Smart Sensors  │
                    │  & Actuators    │
                    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Docker & Docker Compose
- Raspberry Pi (for IoT gateway)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartGardenAI.git
cd SmartGardenAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services with Docker
docker-compose up -d

# Run database migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Start the application
python manage.py runserver
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/smartgarden

# Redis
REDIS_URL=redis://localhost:6379

# API Keys
OPENWEATHER_API_KEY=your_openweather_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token

# IoT Configuration
IOT_GATEWAY_HOST=192.168.1.100
IOT_GATEWAY_PORT=8080

# ML Model Settings
ML_MODEL_PATH=models/plant_health_predictor.h5
MODEL_UPDATE_INTERVAL=3600
```

### IoT Device Setup

```python
# Example sensor configuration
SENSOR_CONFIG = {
    "soil_moisture": {
        "pin": 17,
        "type": "analog",
        "calibration": {"dry": 1023, "wet": 300}
    },
    "temperature": {
        "pin": 18,
        "type": "digital",
        "sensor": "DHT22"
    },
    "light": {
        "pin": 19,
        "type": "analog",
        "sensor": "LDR"
    }
}
```

## API Documentation

### Core Endpoints

```python
# Plant Management
GET    /api/plants/              # List all plants
POST   /api/plants/              # Add new plant
GET    /api/plants/{id}/         # Get plant details
PUT    /api/plants/{id}/         # Update plant
DELETE /api/plants/{id}/         # Remove plant

# Sensor Data
GET    /api/sensors/             # List all sensors
GET    /api/sensors/{id}/data/   # Get sensor readings
POST   /api/sensors/{id}/data/   # Add sensor reading

# AI Predictions
GET    /api/predictions/health/  # Plant health predictions
GET    /api/predictions/water/   # Watering recommendations
POST   /api/predictions/train/   # Retrain ML models

# Automation
GET    /api/automation/status/   # Automation status
POST   /api/automation/water/    # Trigger watering
POST   /api/automation/light/    # Control grow lights
```

### Example API Usage

```python
import requests

# Get plant health predictions
response = requests.get('http://localhost:8000/api/predictions/health/')
predictions = response.json()

# Add sensor reading
sensor_data = {
    "temperature": 24.5,
    "humidity": 65.2,
    "soil_moisture": 0.7,
    "light_intensity": 850
}
requests.post('http://localhost:8000/api/sensors/1/data/', json=sensor_data)
```

## Machine Learning Models

### Plant Health Predictor

```python
from smartgarden.ml.models import PlantHealthPredictor

# Initialize model
model = PlantHealthPredictor()

# Make prediction
features = {
    "temperature": 25.0,
    "humidity": 60.0,
    "soil_moisture": 0.8,
    "light_intensity": 900,
    "plant_age": 30,
    "plant_type": "tomato"
}

health_score = model.predict(features)
print(f"Plant health score: {health_score:.2f}")
```

### Watering Optimization

```python
from smartgarden.ml.optimization import WateringOptimizer

optimizer = WateringOptimizer()

# Get optimal watering schedule
schedule = optimizer.optimize_schedule(
    plant_type="tomato",
    weather_forecast=weather_data,
    soil_conditions=soil_data
)

print(f"Recommended watering: {schedule['frequency']} times per week")
```

## IoT Device Code

### Raspberry Pi Gateway

```python
# iot_gateway/main.py
import asyncio
from smartgarden.iot.gateway import IoTGateway
from smartgarden.sensors import SensorManager

async def main():
    # Initialize gateway
    gateway = IoTGateway()
    
    # Set up sensors
    sensors = SensorManager()
    sensors.add_sensor("soil_moisture", pin=17)
    sensors.add_sensor("temperature", pin=18)
    sensors.add_sensor("light", pin=19)
    
    # Start monitoring
    await gateway.start_monitoring(sensors)

if __name__ == "__main__":
    asyncio.run(main())
```

### Sensor Interface

```python
# sensors/soil_moisture.py
import time
import board
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

class SoilMoistureSensor:
    def __init__(self, pin):
        self.i2c = board.I2C()
        self.ads = ADS.ADS1115(self.i2c)
        self.channel = AnalogIn(self.ads, pin)
        
    def read(self):
        # Read analog value
        raw_value = self.channel.value
        
        # Convert to moisture percentage
        moisture = self._calibrate(raw_value)
        return moisture
    
    def _calibrate(self, raw_value):
        # Calibration curve
        dry_value = 1023
        wet_value = 300
        moisture = ((dry_value - raw_value) / (dry_value - wet_value)) * 100
        return max(0, min(100, moisture))
```

## Mobile App

### React Native Components

```javascript
// App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Dashboard from './screens/Dashboard';
import PlantDetail from './screens/PlantDetail';
import Settings from './screens/Settings';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Dashboard" component={Dashboard} />
        <Stack.Screen name="PlantDetail" component={PlantDetail} />
        <Stack.Screen name="Settings" component={Settings} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

### Dashboard Component

```javascript
// screens/Dashboard.js
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { LineChart, BarChart } from 'react-native-chart-kit';
import { PlantCard, SensorCard, AlertCard } from '../components';

export default function Dashboard() {
  const [plants, setPlants] = useState([]);
  const [sensorData, setSensorData] = useState({});
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/dashboard/');
      const data = await response.json();
      setPlants(data.plants);
      setSensorData(data.sensors);
      setAlerts(data.alerts);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Smart Garden Dashboard</Text>
      
      <View style={styles.sensorGrid}>
        <SensorCard 
          title="Temperature" 
          value={sensorData.temperature} 
          unit="°C" 
        />
        <SensorCard 
          title="Humidity" 
          value={sensorData.humidity} 
          unit="%" 
        />
        <SensorCard 
          title="Soil Moisture" 
          value={sensorData.soil_moisture} 
          unit="%" 
        />
        <SensorCard 
          title="Light" 
          value={sensorData.light_intensity} 
          unit="lux" 
        />
      </View>

      <View style={styles.plantsSection}>
        <Text style={styles.sectionTitle}>Your Plants</Text>
        {plants.map(plant => (
          <PlantCard key={plant.id} plant={plant} />
        ))}
      </View>

      <View style={styles.alertsSection}>
        <Text style={styles.sectionTitle}>Alerts</Text>
        {alerts.map(alert => (
          <AlertCard key={alert.id} alert={alert} />
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 20,
  },
  sensorGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    padding: 10,
  },
  plantsSection: {
    padding: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  alertsSection: {
    padding: 10,
  },
});
```

## Data Analytics

### Dashboard Analytics

```python
# analytics/dashboard.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from smartgarden.models import SensorReading, Plant

class DashboardAnalytics:
    def __init__(self):
        self.db = Database()
    
    def get_sensor_trends(self, days=7):
        """Get sensor data trends over time"""
        query = """
        SELECT timestamp, sensor_type, value
        FROM sensor_readings
        WHERE timestamp >= NOW() - INTERVAL '%s days'
        ORDER BY timestamp
        """ % days
        
        df = pd.read_sql(query, self.db.connection)
        
        # Create trend plots
        fig = px.line(df, x='timestamp', y='value', 
                     color='sensor_type', title='Sensor Trends')
        return fig
    
    def get_plant_health_summary(self):
        """Get plant health summary"""
        plants = Plant.objects.all()
        
        health_data = []
        for plant in plants:
            health_score = plant.get_health_score()
            health_data.append({
                'name': plant.name,
                'health_score': health_score,
                'status': 'Healthy' if health_score > 0.7 else 'Needs Attention'
            })
        
        return pd.DataFrame(health_data)
    
    def get_water_usage_analysis(self):
        """Analyze water usage patterns"""
        query = """
        SELECT DATE(timestamp) as date, SUM(water_volume) as total_water
        FROM irrigation_events
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        df = pd.read_sql(query, self.db.connection)
        
        fig = px.bar(df, x='date', y='total_water', 
                    title='Daily Water Usage')
        return fig
```

## Testing

### Unit Tests

```python
# tests/test_ml_models.py
import unittest
import numpy as np
from smartgarden.ml.models import PlantHealthPredictor

class TestPlantHealthPredictor(unittest.TestCase):
    def setUp(self):
        self.model = PlantHealthPredictor()
    
    def test_prediction_range(self):
        """Test that predictions are in valid range [0, 1]"""
        features = {
            "temperature": 25.0,
            "humidity": 60.0,
            "soil_moisture": 0.8,
            "light_intensity": 900,
            "plant_age": 30,
            "plant_type": "tomato"
        }
        
        prediction = self.model.predict(features)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        importance = self.model.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from smartgarden.main import app

client = TestClient(app)

def test_get_plants():
    response = client.get("/api/plants/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_plant():
    plant_data = {
        "name": "Test Tomato",
        "plant_type": "tomato",
        "location": "garden_bed_1"
    }
    response = client.post("/api/plants/", json=plant_data)
    assert response.status_code == 201
    assert response.json()["name"] == "Test Tomato"

def test_sensor_data_endpoint():
    response = client.get("/api/sensors/1/data/")
    assert response.status_code == 200
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "smartgarden.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/smartgarden
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=smartgarden
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  iot-gateway:
    build: ./iot_gateway
    environment:
      - API_URL=http://web:8000
    volumes:
      - /dev:/dev
    privileged: true

volumes:
  postgres_data:
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with FastAPI, React Native, and TensorFlow
- IoT sensors powered by Raspberry Pi
- Weather data from OpenWeatherMap API
- Icons from Feather Icons 
