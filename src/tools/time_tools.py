# must be updated to use new tool structure
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, Optional, Any
import requests
# Remove dateutil dependency
import json
import re  # Add for regex-based time parsing

# Make geopy optional
try:
    from geopy.geocoders import Nominatim
    has_geopy = True
except ImportError:
    has_geopy = False

from bson import ObjectId

from src.tools.base import (
    BaseTool,
    AgentResult,
    AgentDependencies,
    ToolRegistry
)
# Defer client import to __init__
# from src.clients.time_api_client import TimeApiClient 
from src.services.llm_service import LLMService, ModelType
from src.db.enums import OperationStatus, ToolOperationState, ContentType, ToolType
from src.prompts.tool_prompts import ToolPrompts

logger = logging.getLogger(__name__)

class TimeTool(BaseTool):
    """Tool for handling time-related operations"""
    
    # Static tool configuration
    name = "time"  # Match the ToolType.TIME value exactly
    description = "Tool for time and timezone operations"
    version = "1.0.0"
    
    # Tool registry configuration - optimized for one-shot usage
    registry = ToolRegistry(
        content_type=ContentType.CALENDAR_EVENT,  # Closest match for time data
        tool_type=ToolType.TIME,  # Need to add this to enums
        requires_approval=False,  # No approval needed 
        requires_scheduling=False,  # No scheduling needed
        required_clients=[],
        required_managers=["tool_state_manager"]
    )
    
    def __init__(self, deps: Optional[AgentDependencies] = None):
        """Initialize time tool with dependencies"""
        super().__init__()
        self.deps = deps or AgentDependencies()
        
        # Services will be injected by orchestrator based on registry requirements
        self.tool_state_manager = None
        self.llm_service = None
        self.db = None
        
        # Specific clients for this tool - Import and instantiate here
        self.client = None
        try:
            from src.clients.time_api_client import TimeApiClient # Import moved here
            self.client = TimeApiClient("https://timeapi.io")
            logger.info("TimeApiClient initialized successfully.")
        except ImportError:
            logger.error("Failed to import TimeApiClient. TimeTool may not function correctly.")
        except Exception as e:
            logger.error(f"Error initializing TimeApiClient: {e}")
            
        self.backup_api = "https://worldtimeapi.org/api/timezone"
        
        # Only initialize geolocator if geopy is available
        self.geolocator = None
        if has_geopy:
            try:
                self.geolocator = Nominatim(user_agent="rin_time_bot") # Use unique agent name
                logger.info("Geopy Nominatim geolocator initialized.")
            except Exception as e:
                logger.error(f"Error initializing Geolocator: {e}")
                self.geolocator = None
        
    def inject_dependencies(self, **services):
        """Inject required services - called by orchestrator during registration"""
        self.tool_state_manager = services.get("tool_state_manager")
        self.llm_service = services.get("llm_service")
        self.db = self.tool_state_manager.db if self.tool_state_manager else None
        
    async def run(self, input_data: str) -> Dict:
        """Run the time tool - optimized for one-shot use without approval flow"""
        try:
            # Get or create operation
            operation = await self.tool_state_manager.get_operation(self.deps.session_id)
            if not operation:
                # Create a new operation in COLLECTING state
                operation = await self.tool_state_manager.start_operation(
                    session_id=self.deps.session_id,
                    tool_type=self.registry.tool_type.value,
                    initial_data={"command": input_data},
                    initial_state=ToolOperationState.COLLECTING.value
                )
            
            # Analyze command to extract parameters
            command_analysis = await self._analyze_command(input_data)
            logger.info(f"Time tool analysis: {command_analysis}")
            
            # Generate content (fetch time data)
            content_result = await self._generate_content(
                timezone=command_analysis.get("timezone"),
                action=command_analysis.get("action", "get_time"),
                source_timezone=command_analysis.get("source_timezone"),
                source_time=command_analysis.get("source_time"),
                tool_operation_id=str(operation["_id"]),
                topic=input_data,
                analyzed_params=command_analysis
            )
            
            # Update operation with content result
            await self.tool_state_manager.update_operation(
                session_id=self.deps.session_id,
                tool_operation_id=str(operation["_id"]),
                input_data={
                    "command": input_data,
                    "command_info": command_analysis
                },
                content_updates={
                    "items": content_result.get("items", [])
                }
            )
            
            # Move directly to COMPLETED state for one-shot tools
            await self.tool_state_manager.update_operation(
                session_id=self.deps.session_id,
                tool_operation_id=str(operation["_id"]),
                state=ToolOperationState.COMPLETED.value
            )
            
            # End operation with success
            await self.tool_state_manager.end_operation(
                session_id=self.deps.session_id,
                tool_operation_id=str(operation["_id"]),
                success=True,
                api_response=content_result
            )
            
            # Get the formatted response from content_result
            time_response = content_result.get("response", "I couldn't retrieve the time information.")
            
            return {
                "status": "completed",
                "state": ToolOperationState.COMPLETED.value,
                "response": time_response,
                "requires_chat_response": True,
                "data": content_result.get("data", {})
            }
            
        except Exception as e:
            logger.error(f"Error in time tool: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "response": f"Sorry, I couldn't retrieve time information: {str(e)}",
                "requires_chat_response": True
            }

    async def _analyze_command(self, command: str) -> Dict:
        """Analyze command to extract location/timezone and action"""
        try:
            action = "get_time"  # Default action
            target_timezone = "America/New_York"  # Default timezone (renamed from timezone)
            source_timezone = None
            source_time = None
            
            # Check for conversion intent
            conversion_indicators = ["convert", "difference", "between", "from", "to"]
            if any(indicator in command.lower() for indicator in conversion_indicators):
                action = "convert_time"
                
                # Simple regex-based extraction
                import re
                from_to_match = re.search(r'from\s+([a-zA-Z\s/_-]+)\s+to\s+([a-zA-Z\s/_-]+)', command, re.IGNORECASE)
                if from_to_match:
                    source_timezone = from_to_match.group(1).strip()
                    target_timezone = from_to_match.group(2).strip() # Renamed
                
                time_match = re.search(r'(\d{1,2}:\d{2}(?:\s*[ap]m)?)', command, re.IGNORECASE)
                if time_match:
                    source_time = time_match.group(1)
                else:
                    source_time = datetime.now().strftime("%H:%M")
            else:
                # Get timezone from location in command
                location_indicators = ["in", "at", "for", "time in", "time at", "time for"]
                for indicator in location_indicators:
                    if indicator in command.lower():
                        parts = command.lower().split(indicator, 1)
                        if len(parts) > 1 and parts[1].strip():
                            target_timezone = parts[1].strip() # Renamed
                            break
            
            return {
                "action": action,
                "timezone": target_timezone, # Keep 'timezone' key for compatibility if needed externally
                "source_timezone": source_timezone,
                "source_time": source_time,
                "item_count": 1 
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time command: {e}")
            return {"action": "get_time", "timezone": "America/New_York", "item_count": 1}

    async def _generate_content(
        self, 
        timezone: str = "America/New_York", # Keep param name for external consistency
        action: str = "get_time",
        source_timezone: Optional[str] = None,
        source_time: Optional[str] = None,
        tool_operation_id: Optional[str] = None,
        topic: Optional[str] = None,
        count: int = 1,
        revision_instructions: Optional[str] = None,
        schedule_id: Optional[str] = None,
        analyzed_params: Optional[Dict] = None
    ) -> Dict:
        """Generate time content. Returns dict with data and user response.
           Does NOT interact with the database directly.
        """
        # Use a different variable name internally to avoid shadowing datetime.timezone
        target_tz_str = timezone 
        source_tz_str = source_timezone
        try:
            # Use analyzed_params if available
            if analyzed_params:
                action = analyzed_params.get("action", action)
                target_tz_str = analyzed_params.get("timezone", target_tz_str) # Update internal var
                source_tz_str = analyzed_params.get("source_timezone", source_tz_str) # Update internal var
                source_time = analyzed_params.get("source_time", source_time)
            
            # First resolve timezone - add early error handling
            resolved_timezone = await self._resolve_timezone(target_tz_str)
            if not resolved_timezone:
                return {
                    "status": "error",
                    "error": f"Could not determine timezone for: {target_tz_str}",
                    "content_to_store": None,
                    "response": f"Sorry, I couldn't determine the timezone for: {target_tz_str}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            # Now proceed with the resolved timezone
            result = None
            if action == "get_time":
                result = await self.get_current_time_in_zone(target_tz_str) # Use internal var
            elif action == "convert_time":
                # Also check source timezone resolution for conversion
                resolved_source_tz = await self._resolve_timezone(source_tz_str) if source_tz_str else None
                if source_tz_str and not resolved_source_tz:
                    return {
                        "status": "error",
                        "error": f"Could not determine source timezone: {source_tz_str}",
                        "content_to_store": None,
                        "response": f"Sorry, I couldn't determine the source timezone: {source_tz_str}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                
                if source_tz_str and source_time:
                    result = await self.convert_time_between_zones(
                        from_zone=source_tz_str, # Use internal var
                        date_time=source_time,
                        to_zone=target_tz_str # Use internal var
                    )
                else:
                    return {
                        "status": "error",
                        "error": "Missing source timezone or time for conversion",
                        "content_to_store": None,
                        "response": "I need both a source timezone and a time to convert!",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
            else:
                 return {
                    "status": "error", 
                    "error": f"Unknown action: {action}", 
                    "content_to_store": None,
                    "response": f"Sorry, I don't know how to handle the time action: {action}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Handle potential errors from helper methods
            if result.get("status") == "error":
                 return {
                    "status": "error", 
                    "error": result.get("response", "Failed to process time request"), 
                    "content_to_store": None,
                    "response": result.get("response", "Sorry, I couldn't process that time request."),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                 }
            
            # Format response for immediate display
            time_response = self._format_time_response(result)
            
            # Prepare the dictionary representing the content to be stored
            content_to_store = {
                "action": action,
                "timezone": target_tz_str, # Store the actual timezone string used
                "source_timezone": source_tz_str,
                "source_time": source_time,
                "result": result, # Contains the actual fetched data
                "timestamp": datetime.now(timezone.utc).isoformat() # Now timezone.utc works!
            }
            
            return {
                "status": "success",
                "data": result, # Raw data from fetch
                "content_to_store": content_to_store, # Data for orchestrator to store
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": time_response  # User-facing formatted response
            }

        except Exception as e:
            logger.error(f"Error generating time data: {str(e)}", exc_info=True) # Added exc_info
            return {
                "status": "error", "error": str(e),
                "content_to_store": None,
                "response": f"Sorry, an unexpected error occurred while handling time: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_current_time_in_zone(self, location_or_timezone: str) -> Dict:
        """Get current time for a location or timezone"""
        try:
            # Use tz_str internally
            tz_str = await self._resolve_timezone(location_or_timezone)
            if not tz_str:
                return {
                    "status": "error",
                    "response": f"Could not determine timezone for: {location_or_timezone}",
                    "requires_tts": True,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Fetch time data using the resolved string
            data = await self._fetch_time_data(tz_str)
            
            if not data:
                return {
                    "status": "error",
                    "response": "Failed to fetch time data",
                    "requires_tts": True,
                    "timestamp": datetime.utcnow().isoformat()
                }

            result = {
                "status": "success",
                "location": location_or_timezone, # Original user input
                "timezone": tz_str, # Resolved timezone string
                "current_time": self._format_time(data.get("dateTime")),
                "day_of_week": data.get("dayOfWeek"),
                "dst_active": data.get("dstActive")
            }
            
            return result

        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return {
                "status": "error",
                "response": f"Error: {str(e)}",
                "requires_tts": True,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def convert_time_between_zones(
        self,
        from_zone: str,
        date_time: str,
        to_zone: str
    ) -> Dict:
        """Convert time between timezones"""
        try:
            # Resolve both timezones if they're locations
            from_timezone = await self._resolve_timezone(from_zone)
            to_timezone = await self._resolve_timezone(to_zone)
            
            if not from_timezone or not to_timezone:
                return {
                    "status": "error",
                    "response": "Could not resolve one or both timezones",
                    "requires_tts": True,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Parse the input time
            parsed_time = self._parse_user_time(date_time)
            if not parsed_time:
                return {
                    "status": "error",
                    "response": f"Could not parse time format: {date_time}",
                    "requires_tts": True,
                    "timestamp": datetime.utcnow().isoformat()
                }

            data = await self.client.convert_time_zone(
                from_zone=from_timezone,
                date_time=parsed_time.isoformat(),
                to_zone=to_timezone
            )

            result = {
                "status": "success",
                "from_location": from_zone,
                "to_location": to_zone,
                "from_time": self._format_time(parsed_time.isoformat()),
                "converted_time": self._format_time(data.get("convertedDateTime")),
                "from_timezone": from_timezone,
                "to_timezone": to_timezone
            }
            
            return result

        except Exception as e:
            logger.error(f"Error converting time: {e}")
            return {
                "status": "error",
                "response": f"Error: {str(e)}",
                "requires_tts": True,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _resolve_timezone(self, location_or_timezone: str) -> Optional[str]:
        """Resolve timezone from location using multiple fallbacks"""
        try:
            logger.info(f"Resolving timezone for: {location_or_timezone}")
            
            # Common timezone mappings with variants
            timezone_mappings = {
                # Asia
                "tokyo": "Asia/Tokyo",
                "toyko": "Asia/Tokyo",  # Common misspelling
                "beijing": "Asia/Shanghai",
                "shanghai": "Asia/Shanghai",
                "hong kong": "Asia/Hong_Kong",
                "hongkong": "Asia/Hong_Kong",
                "singapore": "Asia/Singapore",
                "dubai": "Asia/Dubai",
                
                # Europe
                "london": "Europe/London",
                "paris": "Europe/Paris",
                "berlin": "Europe/Berlin",
                "moscow": "Europe/Moscow",
                "amsterdam": "Europe/Amsterdam",
                "rome": "Europe/Rome",
                
                # Americas
                "new york": "America/New_York",
                "nyc": "America/New_York",
                "los angeles": "America/Los_Angeles",
                "la": "America/Los_Angeles",
                "chicago": "America/Chicago",
                "toronto": "America/Toronto",
                
                # Australia/Pacific
                "sydney": "Australia/Sydney",
                "melbourne": "Australia/Melbourne",
                "auckland": "Pacific/Auckland"
            }
            
            # First check if it's already a valid timezone
            if "/" in location_or_timezone:
                logger.info(f"Valid timezone format detected: {location_or_timezone}")
                return location_or_timezone
                
            # Clean and check location
            location_lower = location_or_timezone.lower().strip()
            logger.info(f"Cleaned location: {location_lower}")
            
            # Direct mapping check
            if location_lower in timezone_mappings:
                timezone = timezone_mappings[location_lower]
                logger.info(f"Found direct mapping: {timezone}")
                return timezone
                
            # Try partial matches
            for key, value in timezone_mappings.items():
                if key in location_lower or location_lower in key:
                    logger.info(f"Found partial match: {value} for {location_lower}")
                    return value
            
            # Try geolocation if available
            if self.geolocator:
                try:
                    location = self.geolocator.geocode(location_or_timezone)
                    if location:
                        # This is a simplified approach - in real production code
                        # you would use the coordinates to determine the timezone
                        # For now, if we find a location, use a reasonable default
                        # based on its coordinates (this is a placeholder)
                        logger.info(f"Found location via geocoding: {location}")
                        if location.longitude > 0:  # Eastern hemisphere
                            return "Europe/London"
                        else:
                            return "America/New_York"
                except Exception as geo_error:
                    logger.warning(f"Geocoding failed: {geo_error}")
            
            # If no mapping found, try backup API
            logger.info("No mapping found, trying backup API...")
            try:
                backup_response = requests.get(
                    f"{self.backup_api}/{location_or_timezone}",
                    timeout=5
                )
                
                if backup_response.status_code == 200:
                    timezone = backup_response.json().get("timezone")
                    logger.info(f"Found timezone from API: {timezone}")
                    return timezone
                    
            except Exception as backup_error:
                logger.warning(f"Backup API failed: {backup_error}")
            
            logger.warning(f"Could not resolve timezone for: {location_or_timezone}")
            return None

        except Exception as e:
            logger.error(f"Error resolving timezone: {e}")
            return None

    def _parse_user_time(self, time_str: str) -> Optional[datetime]:
        """Parse various time formats using regex patterns instead of dateutil"""
        try:
            # Try to parse common time formats
            time_str = time_str.strip().lower()
            
            # Handle "HH:MM" format (24-hour)
            if re.match(r'^\d{1,2}:\d{2}$', time_str):
                hour, minute = map(int, time_str.split(':'))
                now = datetime.now()
                return datetime(now.year, now.month, now.day, hour, minute)
            
            # Handle "HH:MM AM/PM" format (12-hour)
            ampm_match = re.match(r'(\d{1,2}):(\d{2})\s*(am|pm)', time_str)
            if ampm_match:
                hour, minute, ampm = ampm_match.groups()
                hour = int(hour)
                minute = int(minute)
                
                # Convert to 24-hour format
                if ampm == 'pm' and hour < 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
                    
                now = datetime.now()
                return datetime(now.year, now.month, now.day, hour, minute)
            
            # Try direct datetime parsing as a fallback
            try:
                return datetime.fromisoformat(time_str)
            except ValueError:
                pass
                
            # Try basic format 
            try:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
                
            # More fallback formats can be added here
            
            # Default fallback - return current time
            logger.warning(f"Could not parse time format: {time_str}, using current time")
            return datetime.now()
            
        except Exception as e:
            logger.error(f"Error parsing time string: {e}")
            return datetime.now()  # Default to current time on error

    def _format_time(self, timestamp: str) -> str:
        """Format timestamp into human-readable format"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%A, %I:%M %p %Z")
        except Exception:
            return timestamp

    async def _fetch_time_data(self, timezone: str) -> Optional[Dict]:
        """Fetch time data with fallback to backup API"""
        try:
            # Try primary API first if available
            if self.client:
                data = await self.client.get_current_time(timezone)
                if data:
                    return data
                
            # If primary fails or not available, try backup API
            try:
                backup_response = requests.get(f"{self.backup_api}/{timezone}", timeout=5)
                if backup_response.status_code == 200:
                    backup_data = backup_response.json()
                    return {
                        "dateTime": backup_data.get("datetime"),
                        "dayOfWeek": datetime.fromisoformat(
                            backup_data.get("datetime").replace('Z', '+00:00')
                        ).strftime("%A"),
                        "dstActive": backup_data.get("dst")
                    }
            except Exception as backup_error:
                logger.warning(f"Backup API failed: {backup_error}")
                
            # If both APIs fail, generate a basic response
            now = datetime.now()
            return {
                "dateTime": now.isoformat(),
                "dayOfWeek": now.strftime("%A"),
                "dstActive": False,
                "generated": True  # Flag to indicate this is generated
            }
            
        except Exception as e:
            logger.error(f"Error fetching time data: {e}")
            # Return minimal data if all else fails
            now = datetime.now()
            return {
                "dateTime": now.isoformat(),
                "dayOfWeek": now.strftime("%A"),
                "dstActive": False,
                "generated": True  # Flag to indicate this is generated
            }

    def _format_time_response(self, result: Dict) -> str:
        """Format time data into human readable response with emojis"""
        if result.get("status") == "error":
            return f"Sorry, {result.get('message', 'an error occurred')}"
            
        if "current_time" in result:
            return f"🕐 The current time in {result['location']} is {result['current_time']}"
        elif "converted_time" in result:
            return f"🕐 When it's {result['from_time']} in {result['from_location']}, it's {result['converted_time']} in {result['to_location']}"
        
        return "I couldn't process that time request"
        
    def can_handle(self, command_text: str, tool_type: Optional[str] = None) -> bool:
        """Check if this tool can handle the given command"""
        # If tool_type is explicitly specified as 'time', handle it
        if tool_type and tool_type.lower() == self.registry.tool_type.value.lower():
            return True
        
        # Otherwise, don't try to detect keywords here - that's the trigger detector's job
        return False