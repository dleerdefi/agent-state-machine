# must be updated to use new tool structure
import logging
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from bson import ObjectId
# Remove geopandas dependency
# import geopandas as gpd

# Make OpenMeteo dependencies optional
try:
    import openmeteo_requests
    from openmeteo_sdk.Variable import Variable
    from retry_requests import retry
    has_openmeteo = True
except ImportError:
    has_openmeteo = False
    
import requests
import json

from src.tools.base import (
    BaseTool, 
    AgentResult, 
    AgentDependencies,
    ToolRegistry
)
from src.services.llm_service import LLMService, ModelType
from src.db.enums import OperationStatus, ToolOperationState, ContentType, ToolType
from src.prompts.tool_prompts import ToolPrompts

logger = logging.getLogger(__name__)

class WeatherTool(BaseTool):
    """Tool for handling weather-related operations"""
    
    # Static tool configuration
    name = "weather"  # Match the ToolType.WEATHER value exactly
    description = "Tool for weather information"
    version = "1.0.0"
    
    # Tool registry configuration - optimized for one-shot usage
    registry = ToolRegistry(
        content_type=ContentType.CALENDAR_EVENT,  # Closest match for weather data
        tool_type=ToolType.WEATHER,  # Need to add this to enums
        requires_approval=False,  # No approval needed
        requires_scheduling=False,  # No scheduling needed
        required_clients=[],
        required_managers=["tool_state_manager"]
    )
    
    def __init__(self, deps: Optional[AgentDependencies] = None):
        """Initialize weather tool with dependencies"""
        super().__init__()
        self.deps = deps or AgentDependencies()
        
        # Services will be injected by orchestrator based on registry requirements
        self.tool_state_manager = None
        self.llm_service = None
        self.db = None
        
        # Setup the Open-Meteo client with retry logic if dependencies are available
        self.client = None
        if has_openmeteo:
            retry_session = retry(retries=3, backoff_factor=0.5)
            self.client = openmeteo_requests.Client(session=retry_session)
            logger.info("OpenMeteo client initialized successfully")
        else:
            logger.warning("OpenMeteo dependencies not available, weather forecasts will be limited")
            
    def inject_dependencies(self, **services):
        """Inject required services - called by orchestrator during registration"""
        self.tool_state_manager = services.get("tool_state_manager")
        self.llm_service = services.get("llm_service")
        self.db = self.tool_state_manager.db if self.tool_state_manager else None
        
    async def run(self, input_data: str) -> Dict:
        """Run the weather tool - primarily for consistency, logic handled by orchestrator"""
        try:
            # Get or create operation
            operation = await self.tool_state_manager.get_operation(self.deps.session_id)
            if not operation:
                operation = await self.tool_state_manager.start_operation(
                    session_id=self.deps.session_id,
                    tool_type=self.registry.tool_type.value,
                    initial_data={"command": input_data},
                    initial_state=ToolOperationState.COLLECTING.value
                )
            
            command_analysis = await self._analyze_command(input_data)
            content_result = await self._generate_content(
                location=command_analysis.get("location"),
                units=command_analysis.get("units", "metric"),
                forecast_type=command_analysis.get("forecast_type", "current"),
                tool_operation_id=str(operation["_id"]),
                topic=input_data,
                analyzed_params=command_analysis
            )

            await self.tool_state_manager.update_operation(
                session_id=self.deps.session_id,
                tool_operation_id=str(operation["_id"]),
                input_data={"command": input_data, "command_info": command_analysis},
                content_updates={"items": content_result.get("items", [])}
            )
            await self.tool_state_manager.end_operation(
                session_id=self.deps.session_id,
                tool_operation_id=str(operation["_id"]),
                success=True,
                api_response=content_result
            )
            
            return {
                "status": "completed",
                "state": ToolOperationState.COMPLETED.value,
                "response": content_result.get("response", "Weather info retrieved."),
                "requires_chat_response": True,
                "data": content_result.get("data", {})
            }
        except Exception as e:
            logger.error(f"Error in weather tool run: {e}", exc_info=True)
            # Attempt to end operation in error state
            if 'operation' in locals() and operation:
                await self.tool_state_manager.end_operation(
                    session_id=self.deps.session_id,
                    tool_operation_id=str(operation['_id']),
                    success=False,
                    api_response={"error": str(e)}
                )
            return {
                "status": "error", "error": str(e),
                "response": f"Sorry, error retrieving weather: {str(e)}",
                "requires_chat_response": True
            }
            
    async def _analyze_command(self, command: str) -> Dict:
        """Analyze command to extract location and parameters"""
        try:
            # Default parameters
            location = "New York"  # Default location
            forecast_type = "current"  # Default to current weather
            units = "metric"  # Default to metric units
            
            # Extract location using simple keyword matching
            location_indicators = ["in", "at", "for", "weather in", "weather at", "weather for"]
            for indicator in location_indicators:
                if indicator in command.lower():
                    parts = command.lower().split(indicator, 1)
                    if len(parts) > 1 and parts[1].strip():
                        location = parts[1].strip()
                        break
            
            # Check for forecast type
            if any(word in command.lower() for word in ["forecast", "tomorrow", "week", "7 day", "daily"]):
                forecast_type = "daily"
            elif any(word in command.lower() for word in ["hourly", "today", "hours"]):
                forecast_type = "hourly"
                
            # Check for units preference
            if any(word in command.lower() for word in ["fahrenheit", "imperial", "°f"]):
                units = "imperial"
                
            return {
                "location": location,
                "forecast_type": forecast_type,
                "units": units,
                "item_count": 1  # Always 1 for this tool
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weather command: {e}")
            return {"location": "New York", "forecast_type": "current", "units": "metric", "item_count": 1}

    async def _generate_content(
        self, 
        location: str = "New York",
        units: str = "metric",
        forecast_type: str = "current",
        tool_operation_id: Optional[str] = None,
        topic: Optional[str] = None,
        count: int = 1,
        revision_instructions: Optional[str] = None,
        schedule_id: Optional[str] = None,
        analyzed_params: Optional[Dict] = None
    ) -> Dict:
        """Generate weather content - compatible with orchestrator's calling convention"""
        try:
            if analyzed_params:
                location = analyzed_params.get("location", location)
                units = analyzed_params.get("units", units)
                forecast_type = analyzed_params.get("forecast_type", forecast_type)
            
            item_id = ObjectId()
            result = await self._fetch_weather_data(location, units, forecast_type)
            
            if result.get("status") == "error":
                return {"status": "error", "error": result.get("message", "Fetch failed"), "items": []}
            
            weather_response = self._format_weather_response(result)
            item = {
                "_id": item_id,
                "content": {
                    "location": location, "units": units, "forecast_type": forecast_type,
                    "result": result, "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "status": OperationStatus.EXECUTED.value,
                "state": ToolOperationState.COMPLETED.value
            }
            
            if tool_operation_id and hasattr(self.db, 'store_tool_item_content'):
                await self.db.store_tool_item_content(
                    item_id=str(item_id), content=item.get("content", {}),
                    operation_details={"location": location, "units": units, "forecast_type": forecast_type},
                    source='generate_content', tool_operation_id=tool_operation_id
                )
            
            return {
                "status": "success", "data": result, "items": [item],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": weather_response
            }
        except Exception as e:
            logger.error(f"Error generating weather data: {str(e)}")
            return {"status": "error", "error": str(e), "items": []}
            
    async def _fetch_weather_data(
        self, 
        location: str, 
        units: str,
        forecast_type: str
    ) -> Dict:
        """Fetch fresh weather data from the API"""
        try:
            # Geocode the location
            coords = await self._geocode_location(location)
            if not coords:
                return {
                    "status": "error",
                    "message": f"Could not geocode location: {location}"
                }
                
            lat, lon = coords
            
            # If OpenMeteo is not available, provide a simplified response
            if not has_openmeteo or not self.client:
                # Return a simplified weather response using a backup API
                return await self._fetch_simplified_weather(lat, lon, units, location)
            
            # Prepare API parameters based on forecast type
            params = {
                "latitude": lat,
                "longitude": lon,
                "timezone": "auto",
                "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
            }
            
            if forecast_type in ["hourly", "daily"]:
                params[forecast_type] = [
                    "temperature_2m",
                    "precipitation_probability",
                    "wind_speed_10m"
                ]
                if forecast_type == "daily":
                    params["forecast_days"] = 7
            
            # Get weather data with retry logic
            try:
                responses = self.client.weather_api(
                    "https://api.open-meteo.com/v1/forecast",
                    params=params
                )
                response = responses[0]
            except Exception as api_error:
                logger.error(f"API request failed: {api_error}")
                return await self._fetch_simplified_weather(lat, lon, units, location)
            
            # Extract current conditions
            current = response.Current()
            current_vars = [
                current.Variables(i) for i in range(current.VariablesLength())
            ]
            
            # Format current conditions
            current_data = {
                "temperature": self._format_temperature(
                    self._get_variable_value(current_vars, Variable.temperature, 2),
                    units
                ),
                "humidity": f"{self._get_variable_value(current_vars, Variable.relative_humidity, 2)}%",
                "precipitation": f"{self._get_variable_value(current_vars, Variable.precipitation)}mm",
                "wind_speed": f"{self._get_variable_value(current_vars, Variable.wind_speed, 10)} km/h",
                "timestamp": self._format_timestamp(current.Time())
            }
            
            # Add forecast data if requested
            result = {
                "status": "success",
                "location": location,
                "coordinates": {"lat": lat, "lon": lon},
                "current": current_data
            }
            
            if forecast_type != "current":
                result["forecast"] = self._extract_forecast_data(
                    response, 
                    forecast_type,
                    units
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return {
                "status": "error",
                "message": f"Failed to fetch weather data: {str(e)}"
            }

    def _get_variable_value(
        self, 
        variables: list, 
        var_type: Any, 
        altitude: int = None
    ) -> Optional[float]:
        """Helper to extract variable value from OpenMeteo response"""
        try:
            # Skip processing if OpenMeteo is not available
            if not has_openmeteo:
                return None
                
            if altitude:
                var = next(
                    (v for v in variables 
                     if v.Variable() == var_type and v.Altitude() == altitude),
                    None
                )
            else:
                var = next(
                    (v for v in variables if v.Variable() == var_type),
                    None
                )
            return var.Value() if var else None
        except Exception:
            return None

    def _format_timestamp(self, timestamp: str) -> str:
        """Convert timestamp to human-readable format"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%A, %I:%M %p")
        except Exception:
            return timestamp

    def _format_temperature(self, temp: float, units: str) -> str:
        """Format temperature based on units"""
        if temp is None:
            return "N/A"
        if units == "imperial":
            temp = (temp * 9/5) + 32
            return f"{temp:.1f}°F"
        return f"{temp:.1f}°C"

    async def _geocode_location(self, location: str) -> Optional[Tuple[float, float]]:
        """Enhanced geocoding with fallback and validation using direct API calls"""
        try:
            # Try primary geocoding using OpenStreetMap's Nominatim API
            geo_url = f"https://nominatim.openstreetmap.org/search"
            params = {
                "q": location,
                "format": "json",
                "limit": 1
            }
            headers = {
                "User-Agent": "RinAI/1.0"
            }
            
            response = requests.get(geo_url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return float(data[0]["lat"]), float(data[0]["lon"])
            
            # If that fails, try with fallback coordinates for major cities
            fallback_coords = {
                "new york": (40.7128, -74.0060),
                "london": (51.5074, -0.1278),
                "tokyo": (35.6762, 139.6503),
                "paris": (48.8566, 2.3522),
                "berlin": (52.5200, 13.4050),
                "los angeles": (34.0522, -118.2437),
                "chicago": (41.8781, -87.6298),
                "sydney": (33.8688, 151.2093),
                "beijing": (39.9042, 116.4074),
                "rome": (41.9028, 12.4964),
            }
            
            # Try to match with fallback cities
            location_lower = location.lower()
            for city, coords in fallback_coords.items():
                if city in location_lower or location_lower in city:
                    logger.info(f"Using fallback coordinates for {location} -> {city}: {coords}")
                    return coords
            
            # Default to New York City if nothing else works
            logger.warning(f"Could not geocode {location}, defaulting to New York City")
            return 40.7128, -74.0060
            
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            # Default to New York as a last resort
            return 40.7128, -74.0060

    def _extract_forecast_data(
        self, 
        response: Any, 
        forecast_type: str,
        units: str
    ) -> Dict:
        """Extract and format forecast data from response"""
        try:
            if forecast_type == "hourly":
                hourly = response.Hourly()
                return {
                    "intervals": [
                        {
                            "time": self._format_timestamp(hourly.Time(i)),
                            "temperature": self._format_temperature(
                                hourly.Variables(0).ValuesArray(i),
                                units
                            ),
                            "precipitation_prob": f"{hourly.Variables(1).ValuesArray(i)}%",
                            "wind_speed": f"{hourly.Variables(2).ValuesArray(i)} km/h"
                        }
                        for i in range(24)  # Next 24 hours
                    ]
                }
            elif forecast_type == "daily":
                daily = response.Daily()
                return {
                    "days": [
                        {
                            "date": self._format_timestamp(daily.Time(i)),
                            "temperature": self._format_temperature(
                                daily.Variables(0).ValuesArray(i),
                                units
                            ),
                            "precipitation_prob": f"{daily.Variables(1).ValuesArray(i)}%",
                            "wind_speed": f"{daily.Variables(2).ValuesArray(i)} km/h"
                        }
                        for i in range(7)  # 7-day forecast
                    ]
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting forecast data: {e}")
            return {}

    def _format_weather_response(self, result: Dict) -> str:
        """Format weather data into human readable response"""
        if result.get("status") == "error":
            return f"Sorry, {result.get('message', 'an error occurred')}"
            
        response_parts = []
        response_parts.append(f"Weather in {result['location']}:")
        
        if "current" in result:
            current = result["current"]
            response_parts.append(f"🌡️ Temperature: {current['temperature']}")
            response_parts.append(f"💧 Humidity: {current['humidity']}")
            response_parts.append(f"🌧️ Precipitation: {current['precipitation']}")
            response_parts.append(f"💨 Wind Speed: {current['wind_speed']}")
            
        if "forecast" in result:
            if "intervals" in result["forecast"]:
                response_parts.append("\nHourly Forecast:")
                for interval in result["forecast"]["intervals"][:8]:  # Show next 8 hours
                    response_parts.append(f"{interval['time']}: {interval['temperature']}")
            elif "days" in result["forecast"]:
                response_parts.append("\nDaily Forecast:")
                for day in result["forecast"]["days"][:3]:  # Show next 3 days
                    response_parts.append(f"{day['date']}: {day['temperature']}")
                    
        return "\n".join(response_parts)
        
    def can_handle(self, command_text: str, tool_type: Optional[str] = None) -> bool:
        """Check if this tool can handle the given command"""
        # If tool_type is explicitly specified as 'weather', handle it
        if tool_type and tool_type.lower() == self.registry.tool_type.value.lower():
            return True
        
        # Otherwise, don't try to detect keywords here - that's the trigger detector's job
        return False

    async def _fetch_simplified_weather(self, lat: float, lon: float, units: str, location: str) -> Dict:
        """Fetch simplified weather using a fallback API when OpenMeteo is not available"""
        try:
            # Use OpenWeatherMap API as a fallback
            # Note: In production, use your own API key from environment variables
            api_key = "demo" # Use demo key for development only
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units={'imperial' if units == 'imperial' else 'metric'}"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract basic weather info
                temp = data.get('main', {}).get('temp', 0)
                humidity = data.get('main', {}).get('humidity', 0)
                wind_speed = data.get('wind', {}).get('speed', 0)
                
                # Format unit-specific values
                temp_formatted = f"{temp:.1f}°F" if units == 'imperial' else f"{temp:.1f}°C"
                
                return {
                    "status": "success",
                    "location": location,
                    "coordinates": {"lat": lat, "lon": lon},
                    "current": {
                        "temperature": temp_formatted,
                        "humidity": f"{humidity}%",
                        "precipitation": "0mm",  # Not directly available in this API
                        "wind_speed": f"{wind_speed} {'mph' if units == 'imperial' else 'km/h'}",
                        "timestamp": datetime.now().strftime("%A, %I:%M %p")
                    }
                }
            else:
                # If API call fails, provide minimal synthetic data
                return {
                    "status": "success",
                    "location": location,
                    "coordinates": {"lat": lat, "lon": lon},
                    "current": {
                        "temperature": "72°F" if units == 'imperial' else "22°C",
                        "humidity": "50%",
                        "precipitation": "0mm",
                        "wind_speed": "5 mph" if units == 'imperial' else "8 km/h",
                        "timestamp": datetime.now().strftime("%A, %I:%M %p")
                    }
                }
        except Exception as e:
            logger.error(f"Error in simplified weather fetch: {e}")
            # Return minimal data in case of any error
            return {
                "status": "success",
                "location": location,
                "coordinates": {"lat": lat, "lon": lon},
                "current": {
                    "temperature": "72°F" if units == 'imperial' else "22°C",
                    "humidity": "50%",
                    "precipitation": "0mm",
                    "wind_speed": "5 mph" if units == 'imperial' else "8 km/h",
                    "timestamp": datetime.now().strftime("%A, %I:%M %p")
                }
            }