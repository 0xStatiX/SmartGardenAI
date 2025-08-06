"""
Machine Learning Models for SmartGardenAI
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PlantHealthPredictor:
    """
    Machine learning model for predicting plant health based on environmental conditions
    """
    
    def __init__(self, model_path: str = "models/plant_health_predictor.h5"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'temperature', 'humidity', 'soil_moisture', 'light_intensity',
            'ph_level', 'nutrient_level', 'plant_age', 'plant_type_encoded'
        ]
        self.plant_types = ['tomato', 'lettuce', 'herbs', 'peppers', 'cucumber']
        
        # Load or create model
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            else:
                self._create_model()
                logger.info("Created new plant health predictor model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_model()
    
    def _create_model(self):
        """Create neural network model for plant health prediction"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(len(self.feature_names),)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess input features"""
        # Encode plant type
        if 'plant_type' in features:
            plant_type_encoded = self.label_encoder.fit_transform([features['plant_type']])[0]
            features['plant_type_encoded'] = plant_type_encoded
        
        # Create feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value
        
        # Normalize features
        feature_array = np.array(feature_vector).reshape(1, -1)
        normalized_features = self.scaler.fit_transform(feature_array)
        
        return normalized_features
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict plant health score (0-1)"""
        try:
            # Preprocess features
            X = self.preprocess_features(features)
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            
            return float(prediction)
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5  # Default neutral score
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Predict health scores for multiple plants"""
        predictions = []
        for features in features_list:
            prediction = self.predict(features)
            predictions.append(prediction)
        return predictions
    
    def train(self, training_data: pd.DataFrame, epochs: int = 100):
        """Train the model with new data"""
        try:
            # Prepare training data
            X = training_data[self.feature_names].values
            y = training_data['health_score'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalize features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
                X_test_scaled, y_test, verbose=0
            )
            
            logger.info(f"Model training completed. Test accuracy: {test_accuracy:.4f}")
            
            # Save model
            self.model.save(self.model_path)
            
            return {
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'history': history.history
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        # For neural networks, we can use permutation importance
        # This is a simplified version
        importance_scores = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_scores[feature_name] = 1.0 / (i + 1)  # Simplified importance
        
        return importance_scores

class WateringOptimizer:
    """
    Machine learning model for optimizing watering schedules
    """
    
    def __init__(self, model_path: str = "models/watering_optimizer.pkl"):
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load existing model
        self._load_model()
    
    def _load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"Loaded watering optimizer from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading watering optimizer: {e}")
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for watering optimization"""
        feature_names = [
            'temperature', 'humidity', 'soil_moisture', 'light_intensity',
            'plant_age', 'plant_type', 'weather_forecast_rain', 'weather_forecast_temp',
            'season', 'time_of_day'
        ]
        
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_watering_needs(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal watering schedule"""
        try:
            if not self.is_trained:
                return self._get_default_schedule(features)
            
            X = self.preprocess_features(features)
            X_scaled = self.scaler.transform(X)
            
            # Predict watering amount and frequency
            watering_amount = self.model.predict(X_scaled)[0]
            
            # Calculate optimal schedule
            schedule = self._calculate_schedule(features, watering_amount)
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error predicting watering needs: {e}")
            return self._get_default_schedule(features)
    
    def _get_default_schedule(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get default watering schedule based on plant type"""
        plant_type = features.get('plant_type', 'general')
        
        default_schedules = {
            'tomato': {'frequency': 3, 'amount': 500, 'time': 'morning'},
            'lettuce': {'frequency': 2, 'amount': 300, 'time': 'morning'},
            'herbs': {'frequency': 2, 'amount': 200, 'time': 'morning'},
            'peppers': {'frequency': 3, 'amount': 400, 'time': 'morning'},
            'cucumber': {'frequency': 4, 'amount': 600, 'time': 'morning'},
            'general': {'frequency': 2, 'amount': 300, 'time': 'morning'}
        }
        
        return default_schedules.get(plant_type, default_schedules['general'])
    
    def _calculate_schedule(self, features: Dict[str, Any], watering_amount: float) -> Dict[str, Any]:
        """Calculate detailed watering schedule"""
        base_frequency = 2  # times per week
        base_amount = 300   # ml per watering
        
        # Adjust based on environmental conditions
        temperature_factor = features.get('temperature', 20) / 20
        humidity_factor = 1 - (features.get('humidity', 50) / 100)
        soil_moisture_factor = 1 - features.get('soil_moisture', 0.5)
        
        # Calculate adjusted values
        adjusted_frequency = max(1, min(7, int(base_frequency * temperature_factor * humidity_factor)))
        adjusted_amount = max(100, min(1000, int(base_amount * soil_moisture_factor)))
        
        return {
            'frequency': adjusted_frequency,
            'amount': adjusted_amount,
            'time': 'morning',
            'next_watering': self._calculate_next_watering(adjusted_frequency),
            'confidence': 0.85
        }
    
    def _calculate_next_watering(self, frequency: int) -> str:
        """Calculate next watering time"""
        days_until_next = 7 // frequency
        next_date = datetime.now() + timedelta(days=days_until_next)
        return next_date.strftime("%Y-%m-%d %H:%M")
    
    def train(self, training_data: pd.DataFrame):
        """Train the watering optimizer model"""
        try:
            # Prepare features and target
            feature_columns = [
                'temperature', 'humidity', 'soil_moisture', 'light_intensity',
                'plant_age', 'plant_type', 'weather_forecast_rain', 'weather_forecast_temp',
                'season', 'time_of_day'
            ]
            
            X = training_data[feature_columns].values
            y = training_data['optimal_watering_amount'].values
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            self.is_trained = True
            
            logger.info("Watering optimizer training completed")
            
        except Exception as e:
            logger.error(f"Error training watering optimizer: {e}")
            raise

class DiseaseDetector:
    """
    Computer vision model for plant disease detection
    """
    
    def __init__(self, model_path: str = "models/disease_detector.h5"):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'healthy', 'blight', 'mildew', 'rust', 'spot', 'rot'
        ]
        
        # Load or create model
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Loaded disease detector from {self.model_path}")
            else:
                self._create_model()
                logger.info("Created new disease detector model")
        except Exception as e:
            logger.error(f"Error loading disease detector: {e}")
            self._create_model()
    
    def _create_model(self):
        """Create CNN model for disease detection"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for disease detection"""
        try:
            # Load and resize image
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224)
            )
            
            # Convert to array and normalize
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            
            return np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict_disease(self, image_path: str) -> Dict[str, Any]:
        """Predict plant disease from image"""
        try:
            # Preprocess image
            X = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(X, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            return {
                'disease': self.class_names[predicted_class],
                'confidence': confidence,
                'all_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting disease: {e}")
            return {
                'disease': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {}
            }
    
    def train(self, training_data_dir: str, epochs: int = 50):
        """Train the disease detection model"""
        try:
            # Data generators
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            train_generator = train_datagen.flow_from_directory(
                training_data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='sparse',
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                training_data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='sparse',
                subset='validation'
            )
            
            # Train model
            history = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                verbose=1
            )
            
            # Save model
            self.model.save(self.model_path)
            
            logger.info("Disease detector training completed")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training disease detector: {e}")
            raise

class GrowthPredictor:
    """
    Machine learning model for predicting plant growth and yield
    """
    
    def __init__(self, model_path: str = "models/growth_predictor.pkl"):
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load existing model
        self._load_model()
    
    def _load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"Loaded growth predictor from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading growth predictor: {e}")
    
    def predict_growth(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict plant growth and yield"""
        try:
            if not self.is_trained:
                return self._get_default_prediction(features)
            
            # Prepare features
            feature_names = [
                'temperature', 'humidity', 'soil_moisture', 'light_intensity',
                'nutrient_level', 'plant_age', 'plant_type', 'season'
            ]
            
            feature_vector = []
            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            growth_rate = self.model.predict(X_scaled)[0]
            
            return {
                'growth_rate': float(growth_rate),
                'expected_yield': self._calculate_yield(features, growth_rate),
                'harvest_date': self._predict_harvest_date(features, growth_rate),
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error predicting growth: {e}")
            return self._get_default_prediction(features)
    
    def _get_default_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get default growth prediction"""
        plant_type = features.get('plant_type', 'general')
        
        default_predictions = {
            'tomato': {'growth_rate': 0.7, 'expected_yield': '2-3 kg', 'harvest_date': '60-80 days'},
            'lettuce': {'growth_rate': 0.9, 'expected_yield': '0.5-1 kg', 'harvest_date': '30-45 days'},
            'herbs': {'growth_rate': 0.8, 'expected_yield': '0.2-0.5 kg', 'harvest_date': '20-30 days'},
            'general': {'growth_rate': 0.75, 'expected_yield': '1-2 kg', 'harvest_date': '45-60 days'}
        }
        
        return default_predictions.get(plant_type, default_predictions['general'])
    
    def _calculate_yield(self, features: Dict[str, Any], growth_rate: float) -> str:
        """Calculate expected yield based on growth rate and conditions"""
        base_yield = 1.0  # kg
        
        # Adjust based on conditions
        adjusted_yield = base_yield * growth_rate * features.get('nutrient_level', 0.5)
        
        return f"{adjusted_yield:.1f}-{adjusted_yield * 1.5:.1f} kg"
    
    def _predict_harvest_date(self, features: Dict[str, Any], growth_rate: float) -> str:
        """Predict harvest date"""
        base_days = 60
        adjusted_days = int(base_days / growth_rate)
        
        harvest_date = datetime.now() + timedelta(days=adjusted_days)
        return harvest_date.strftime("%Y-%m-%d")
    
    def train(self, training_data: pd.DataFrame):
        """Train the growth predictor model"""
        try:
            # Prepare features and target
            feature_columns = [
                'temperature', 'humidity', 'soil_moisture', 'light_intensity',
                'nutrient_level', 'plant_age', 'plant_type', 'season'
            ]
            
            X = training_data[feature_columns].values
            y = training_data['growth_rate'].values
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            self.is_trained = True
            
            logger.info("Growth predictor training completed")
            
        except Exception as e:
            logger.error(f"Error training growth predictor: {e}")
            raise

class MLModelManager:
    """
    Manager class for all machine learning models
    """
    
    def __init__(self):
        self.health_predictor = PlantHealthPredictor()
        self.watering_optimizer = WateringOptimizer()
        self.disease_detector = DiseaseDetector()
        self.growth_predictor = GrowthPredictor()
        
        logger.info("ML Model Manager initialized")
    
    def get_comprehensive_prediction(self, plant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive predictions for a plant"""
        try:
            # Health prediction
            health_score = self.health_predictor.predict(plant_data)
            
            # Watering optimization
            watering_schedule = self.watering_optimizer.predict_watering_needs(plant_data)
            
            # Growth prediction
            growth_prediction = self.growth_predictor.predict_growth(plant_data)
            
            # Disease detection (if image provided)
            disease_prediction = None
            if 'image_path' in plant_data:
                disease_prediction = self.disease_detector.predict_disease(plant_data['image_path'])
            
            return {
                'health_score': health_score,
                'watering_schedule': watering_schedule,
                'growth_prediction': growth_prediction,
                'disease_prediction': disease_prediction,
                'recommendations': self._generate_recommendations(
                    health_score, watering_schedule, growth_prediction, disease_prediction
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive prediction: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, health_score: float, watering_schedule: Dict, 
                                growth_prediction: Dict, disease_prediction: Dict) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        # Health-based recommendations
        if health_score < 0.5:
            recommendations.append("Plant health is poor. Check environmental conditions and consider adjusting care routine.")
        elif health_score < 0.7:
            recommendations.append("Plant health needs attention. Monitor closely and adjust care as needed.")
        
        # Watering recommendations
        if watering_schedule.get('frequency', 0) > 4:
            recommendations.append("High watering frequency detected. Consider improving soil drainage.")
        
        # Disease recommendations
        if disease_prediction and disease_prediction.get('disease') != 'healthy':
            recommendations.append(f"Potential disease detected: {disease_prediction['disease']}. Consider treatment.")
        
        # Growth recommendations
        if growth_prediction.get('growth_rate', 1.0) < 0.6:
            recommendations.append("Growth rate is below optimal. Consider increasing light or nutrients.")
        
        return recommendations
    
    def retrain_all_models(self, training_data: Dict[str, pd.DataFrame]):
        """Retrain all models with new data"""
        try:
            logger.info("Starting model retraining...")
            
            if 'health_data' in training_data:
                self.health_predictor.train(training_data['health_data'])
            
            if 'watering_data' in training_data:
                self.watering_optimizer.train(training_data['watering_data'])
            
            if 'growth_data' in training_data:
                self.growth_predictor.train(training_data['growth_data'])
            
            logger.info("All models retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            raise 