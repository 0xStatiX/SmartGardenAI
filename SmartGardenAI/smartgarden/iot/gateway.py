"""
IoT Gateway for SmartGardenAI
Manages sensors, actuators, and device communication
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import paho.mqtt.client as mqtt
from paho.mqtt import publish
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import RPi.GPIO as GPIO
from threading import Thread, Lock

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Data class for sensor readings"""
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime
    location: str
    quality: float = 1.0

@dataclass
class ActuatorCommand:
    """Data class for actuator commands"""
    actuator_id: str
    actuator_type: str
    command: str
    value: Optional[float] = None
    duration: Optional[int] = None
    timestamp: datetime = None

class Sensor:
    """Base class for all sensors"""
    
    def __init__(self, sensor_id: str, sensor_type: str, location: str):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.location = location
        self.last_reading = None
        self.is_active = True
        self.calibration_data = {}
    
    async def read(self) -> Optional[SensorReading]:
        """Read sensor data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def calibrate(self, calibration_data: Dict[str, Any]):
        """Calibrate sensor with provided data"""
        self.calibration_data = calibration_data
    
    def get_status(self) -> Dict[str, Any]:
        """Get sensor status"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'location': self.location,
            'is_active': self.is_active,
            'last_reading': self.last_reading.isoformat() if self.last_reading else None,
            'calibration_data': self.calibration_data
        }

class SoilMoistureSensor(Sensor):
    """Soil moisture sensor implementation"""
    
    def __init__(self, sensor_id: str, location: str, pin: int, adc_channel: int = 0):
        super().__init__(sensor_id, "soil_moisture", location)
        self.pin = pin
        self.adc_channel = adc_channel
        self.adc = None
        self.channel = None
        self._setup_adc()
    
    def _setup_adc(self):
        """Setup ADC for analog reading"""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.adc = ADS.ADS1115(i2c)
            self.channel = AnalogIn(self.adc, self.adc_channel)
            logger.info(f"ADC setup successful for soil moisture sensor {self.sensor_id}")
        except Exception as e:
            logger.error(f"Failed to setup ADC for sensor {self.sensor_id}: {e}")
    
    async def read(self) -> Optional[SensorReading]:
        """Read soil moisture level"""
        try:
            if not self.adc or not self.channel:
                return None
            
            # Read raw ADC value
            raw_value = self.channel.value
            
            # Convert to moisture percentage
            moisture_percentage = self._convert_to_moisture(raw_value)
            
            # Create reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=moisture_percentage,
                unit="%",
                timestamp=datetime.now(),
                location=self.location
            )
            
            self.last_reading = reading
            return reading
            
        except Exception as e:
            logger.error(f"Error reading soil moisture sensor {self.sensor_id}: {e}")
            return None
    
    def _convert_to_moisture(self, raw_value: int) -> float:
        """Convert raw ADC value to moisture percentage"""
        # Default calibration values
        dry_value = self.calibration_data.get('dry_value', 1023)
        wet_value = self.calibration_data.get('wet_value', 300)
        
        # Convert to percentage
        moisture = ((dry_value - raw_value) / (dry_value - wet_value)) * 100
        return max(0.0, min(100.0, moisture))

class TemperatureHumiditySensor(Sensor):
    """Temperature and humidity sensor implementation"""
    
    def __init__(self, sensor_id: str, location: str, pin: int):
        super().__init__(sensor_id, "temperature_humidity", location)
        self.pin = pin
        self.dht = None
        self._setup_dht()
    
    def _setup_dht(self):
        """Setup DHT sensor"""
        try:
            import adafruit_dht
            self.dht = adafruit_dht.DHT22(self.pin)
            logger.info(f"DHT sensor setup successful for {self.sensor_id}")
        except Exception as e:
            logger.error(f"Failed to setup DHT sensor {self.sensor_id}: {e}")
    
    async def read(self) -> Optional[SensorReading]:
        """Read temperature and humidity"""
        try:
            if not self.dht:
                return None
            
            # Read temperature and humidity
            temperature = self.dht.temperature
            humidity = self.dht.humidity
            
            # Create reading (return temperature, humidity can be read separately)
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type="temperature",
                value=temperature,
                unit="°C",
                timestamp=datetime.now(),
                location=self.location
            )
            
            self.last_reading = reading
            return reading
            
        except Exception as e:
            logger.error(f"Error reading temperature sensor {self.sensor_id}: {e}")
            return None

class LightSensor(Sensor):
    """Light intensity sensor implementation"""
    
    def __init__(self, sensor_id: str, location: str, pin: int, adc_channel: int = 1):
        super().__init__(sensor_id, "light", location)
        self.pin = pin
        self.adc_channel = adc_channel
        self.adc = None
        self.channel = None
        self._setup_adc()
    
    def _setup_adc(self):
        """Setup ADC for light sensor"""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.adc = ADS.ADS1115(i2c)
            self.channel = AnalogIn(self.adc, self.adc_channel)
            logger.info(f"ADC setup successful for light sensor {self.sensor_id}")
        except Exception as e:
            logger.error(f"Failed to setup ADC for light sensor {self.sensor_id}: {e}")
    
    async def read(self) -> Optional[SensorReading]:
        """Read light intensity"""
        try:
            if not self.adc or not self.channel:
                return None
            
            # Read raw ADC value
            raw_value = self.channel.value
            
            # Convert to lux
            lux = self._convert_to_lux(raw_value)
            
            # Create reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=lux,
                unit="lux",
                timestamp=datetime.now(),
                location=self.location
            )
            
            self.last_reading = reading
            return reading
            
        except Exception as e:
            logger.error(f"Error reading light sensor {self.sensor_id}: {e}")
            return None
    
    def _convert_to_lux(self, raw_value: int) -> float:
        """Convert raw ADC value to lux"""
        # Simplified conversion - would need proper calibration
        voltage = (raw_value / 32767) * 4.096  # ADS1115 reference voltage
        resistance = (4.096 - voltage) / voltage * 10000  # Voltage divider
        lux = 1000000 / resistance  # Approximate lux calculation
        return max(0.0, lux)

class Actuator:
    """Base class for all actuators"""
    
    def __init__(self, actuator_id: str, actuator_type: str, location: str):
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self.location = location
        self.is_active = True
        self.current_state = "off"
        self.last_command = None
    
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """Execute actuator command - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, Any]:
        """Get actuator status"""
        return {
            'actuator_id': self.actuator_id,
            'actuator_type': self.actuator_type,
            'location': self.location,
            'is_active': self.is_active,
            'current_state': self.current_state,
            'last_command': self.last_command.isoformat() if self.last_command else None
        }

class WaterPump(Actuator):
    """Water pump actuator implementation"""
    
    def __init__(self, actuator_id: str, location: str, pin: int):
        super().__init__(actuator_id, "water_pump", location)
        self.pin = pin
        self.pwm = None
        self._setup_gpio()
    
    def _setup_gpio(self):
        """Setup GPIO for water pump"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pin, 100)  # 100 Hz PWM
            self.pwm.start(0)
            logger.info(f"GPIO setup successful for water pump {self.actuator_id}")
        except Exception as e:
            logger.error(f"Failed to setup GPIO for water pump {self.actuator_id}: {e}")
    
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """Execute water pump command"""
        try:
            if command.command == "start":
                speed = command.value or 100  # Default to 100% speed
                self.pwm.ChangeDutyCycle(speed)
                self.current_state = "on"
                self.last_command = datetime.now()
                logger.info(f"Water pump {self.actuator_id} started at {speed}% speed")
                return True
                
            elif command.command == "stop":
                self.pwm.ChangeDutyCycle(0)
                self.current_state = "off"
                self.last_command = datetime.now()
                logger.info(f"Water pump {self.actuator_id} stopped")
                return True
                
            elif command.command == "pulse":
                duration = command.duration or 5  # Default 5 seconds
                speed = command.value or 100
                
                # Start pump
                self.pwm.ChangeDutyCycle(speed)
                self.current_state = "on"
                
                # Wait for duration
                await asyncio.sleep(duration)
                
                # Stop pump
                self.pwm.ChangeDutyCycle(0)
                self.current_state = "off"
                self.last_command = datetime.now()
                
                logger.info(f"Water pump {self.actuator_id} pulsed for {duration} seconds")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing water pump command: {e}")
            return False

class GrowLight(Actuator):
    """Grow light actuator implementation"""
    
    def __init__(self, actuator_id: str, location: str, pin: int):
        super().__init__(actuator_id, "grow_light", location)
        self.pin = pin
        self._setup_gpio()
    
    def _setup_gpio(self):
        """Setup GPIO for grow light"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
            logger.info(f"GPIO setup successful for grow light {self.actuator_id}")
        except Exception as e:
            logger.error(f"Failed to setup GPIO for grow light {self.actuator_id}: {e}")
    
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """Execute grow light command"""
        try:
            if command.command == "on":
                GPIO.output(self.pin, GPIO.HIGH)
                self.current_state = "on"
                self.last_command = datetime.now()
                logger.info(f"Grow light {self.actuator_id} turned on")
                return True
                
            elif command.command == "off":
                GPIO.output(self.pin, GPIO.LOW)
                self.current_state = "off"
                self.last_command = datetime.now()
                logger.info(f"Grow light {self.actuator_id} turned off")
                return True
                
            elif command.command == "schedule":
                # Implement scheduling logic
                duration = command.duration or 12  # Default 12 hours
                await self._run_schedule(duration)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing grow light command: {e}")
            return False
    
    async def _run_schedule(self, duration: int):
        """Run grow light schedule"""
        try:
            # Turn on light
            GPIO.output(self.pin, GPIO.HIGH)
            self.current_state = "on"
            
            # Wait for duration
            await asyncio.sleep(duration * 3600)  # Convert hours to seconds
            
            # Turn off light
            GPIO.output(self.pin, GPIO.LOW)
            self.current_state = "off"
            
            self.last_command = datetime.now()
            logger.info(f"Grow light {self.actuator_id} schedule completed")
            
        except Exception as e:
            logger.error(f"Error running grow light schedule: {e}")

class IoTGateway:
    """Main IoT Gateway class"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.sensors: Dict[str, Sensor] = {}
        self.actuators: Dict[str, Actuator] = {}
        self.mqtt_client = None
        self.is_running = False
        self.reading_interval = 30  # seconds
        self.data_buffer: List[SensorReading] = []
        self.buffer_lock = Lock()
        
        # Initialize MQTT
        self._setup_mqtt()
        
        # Initialize default sensors and actuators
        self._initialize_devices()
    
    def _setup_mqtt(self):
        """Setup MQTT client for communication"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT client setup successful")
        except Exception as e:
            logger.error(f"Failed to setup MQTT client: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        logger.info(f"MQTT connected with result code {rc}")
        client.subscribe("smartgarden/+/command")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            payload = json.loads(msg.payload.decode())
            asyncio.create_task(self._handle_mqtt_command(payload))
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")
    
    async def _handle_mqtt_command(self, payload: Dict[str, Any]):
        """Handle MQTT command"""
        try:
            actuator_id = payload.get('actuator_id')
            command = payload.get('command')
            value = payload.get('value')
            duration = payload.get('duration')
            
            if actuator_id in self.actuators:
                actuator_command = ActuatorCommand(
                    actuator_id=actuator_id,
                    actuator_type=self.actuators[actuator_id].actuator_type,
                    command=command,
                    value=value,
                    duration=duration,
                    timestamp=datetime.now()
                )
                
                success = await self.actuators[actuator_id].execute_command(actuator_command)
                
                # Publish result
                result = {
                    'actuator_id': actuator_id,
                    'command': command,
                    'success': success,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.mqtt_client.publish(f"smartgarden/{actuator_id}/result", json.dumps(result))
                
        except Exception as e:
            logger.error(f"Error handling MQTT command: {e}")
    
    def _initialize_devices(self):
        """Initialize default sensors and actuators"""
        try:
            # Add sensors
            self.add_sensor(SoilMoistureSensor("soil_1", "garden_bed_1", 17, 0))
            self.add_sensor(SoilMoistureSensor("soil_2", "garden_bed_2", 17, 1))
            self.add_sensor(TemperatureHumiditySensor("temp_hum_1", "garden_bed_1", 18))
            self.add_sensor(LightSensor("light_1", "garden_bed_1", 19, 2))
            
            # Add actuators
            self.add_actuator(WaterPump("pump_1", "garden_bed_1", 20))
            self.add_actuator(WaterPump("pump_2", "garden_bed_2", 21))
            self.add_actuator(GrowLight("light_1", "garden_bed_1", 22))
            
            logger.info("Default devices initialized")
            
        except Exception as e:
            logger.error(f"Error initializing devices: {e}")
    
    def add_sensor(self, sensor: Sensor):
        """Add a sensor to the gateway"""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Added sensor: {sensor.sensor_id}")
    
    def add_actuator(self, actuator: Actuator):
        """Add an actuator to the gateway"""
        self.actuators[actuator.actuator_id] = actuator
        logger.info(f"Added actuator: {actuator.actuator_id}")
    
    async def start_monitoring(self):
        """Start monitoring all sensors"""
        self.is_running = True
        logger.info("Starting IoT gateway monitoring")
        
        while self.is_running:
            try:
                # Read all sensors
                readings = []
                for sensor in self.sensors.values():
                    if sensor.is_active:
                        reading = await sensor.read()
                        if reading:
                            readings.append(reading)
                
                # Add to buffer
                with self.buffer_lock:
                    self.data_buffer.extend(readings)
                
                # Send data to API if buffer is full
                if len(self.data_buffer) >= 10:
                    await self._send_data_to_api()
                
                # Publish to MQTT
                for reading in readings:
                    self.mqtt_client.publish(
                        f"smartgarden/sensor/{reading.sensor_id}",
                        json.dumps(asdict(reading))
                    )
                
                # Wait for next reading
                await asyncio.sleep(self.reading_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        logger.info("Stopping IoT gateway monitoring")
        
        # Cleanup GPIO
        try:
            GPIO.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up GPIO: {e}")
    
    async def _send_data_to_api(self):
        """Send buffered data to API"""
        try:
            with self.buffer_lock:
                data_to_send = self.data_buffer.copy()
                self.data_buffer.clear()
            
            if not data_to_send:
                return
            
            # Convert to API format
            api_data = []
            for reading in data_to_send:
                api_data.append({
                    'sensor_id': reading.sensor_id,
                    'sensor_type': reading.sensor_type,
                    'value': reading.value,
                    'unit': reading.unit,
                    'timestamp': reading.timestamp.isoformat(),
                    'location': reading.location
                })
            
            # Send to API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/v1/sensors/data/batch",
                    json=api_data
                ) as response:
                    if response.status == 200:
                        logger.info(f"Sent {len(api_data)} sensor readings to API")
                    else:
                        logger.error(f"Failed to send data to API: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending data to API: {e}")
    
    async def execute_actuator_command(self, actuator_id: str, command: str, 
                                     value: Optional[float] = None, 
                                     duration: Optional[int] = None) -> bool:
        """Execute command on specific actuator"""
        try:
            if actuator_id not in self.actuators:
                logger.error(f"Actuator {actuator_id} not found")
                return False
            
            actuator_command = ActuatorCommand(
                actuator_id=actuator_id,
                actuator_type=self.actuators[actuator_id].actuator_type,
                command=command,
                value=value,
                duration=duration,
                timestamp=datetime.now()
            )
            
            success = await self.actuators[actuator_id].execute_command(actuator_command)
            return success
            
        except Exception as e:
            logger.error(f"Error executing actuator command: {e}")
            return False
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive gateway status"""
        sensor_statuses = {sensor_id: sensor.get_status() 
                          for sensor_id, sensor in self.sensors.items()}
        
        actuator_statuses = {actuator_id: actuator.get_status() 
                           for actuator_id, actuator in self.actuators.items()}
        
        return {
            'gateway_id': 'main_gateway',
            'is_running': self.is_running,
            'api_url': self.api_url,
            'reading_interval': self.reading_interval,
            'buffer_size': len(self.data_buffer),
            'sensors': sensor_statuses,
            'actuators': actuator_statuses,
            'mqtt_connected': self.mqtt_client.is_connected() if self.mqtt_client else False,
            'timestamp': datetime.now().isoformat()
        }
    
    async def calibrate_sensor(self, sensor_id: str, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate a specific sensor"""
        try:
            if sensor_id not in self.sensors:
                logger.error(f"Sensor {sensor_id} not found")
                return False
            
            self.sensors[sensor_id].calibrate(calibration_data)
            logger.info(f"Sensor {sensor_id} calibrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calibrating sensor {sensor_id}: {e}")
            return False
    
    async def get_sensor_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get current reading from specific sensor"""
        try:
            if sensor_id not in self.sensors:
                logger.error(f"Sensor {sensor_id} not found")
                return None
            
            return await self.sensors[sensor_id].read()
            
        except Exception as e:
            logger.error(f"Error reading sensor {sensor_id}: {e}")
            return None

# Example usage
async def main():
    """Example usage of IoT Gateway"""
    gateway = IoTGateway()
    
    try:
        # Start monitoring
        await gateway.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Shutting down gateway...")
    finally:
        await gateway.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 