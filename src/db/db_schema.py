from typing import TypedDict, List, Optional, Dict, Literal, Any, Union
from datetime import datetime, timezone
import logging
import uuid
from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from src.db.enums import (
    OperationStatus,
    ToolOperationState,
    ScheduleState,
    ApprovalState,
    ContentType,
    ToolType
)
from enum import Enum
from pydantic import BaseModel
from functools import wraps
import json

logger = logging.getLogger(__name__)

def db_operation_logger(operation_name: str):
    """Decorator for logging database operations"""
    def decorator(func: callable) -> callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Log operation start
                logger.info(f"Starting {operation_name}")
                
                # Log parameters (safely)
                safe_args = [
                    str(arg) if isinstance(arg, ObjectId) 
                    else json.dumps(arg) if isinstance(arg, dict)
                    else str(arg)
                    for arg in args[1:]  # Skip self
                ]
                safe_kwargs = {
                    k: str(v) if isinstance(v, ObjectId)
                    else json.dumps(v) if isinstance(v, dict)
                    else str(v)
                    for k, v in kwargs.items()
                }
                logger.info(f"Parameters - args: {safe_args}, kwargs: {safe_kwargs}")
                
                # Execute operation
                result = await func(*args, **kwargs)
                
                # Log success
                if result:
                    if isinstance(result, dict):
                        logger.info(f"Successfully completed {operation_name}")
                        logger.info(f"Updated fields: {list(result.keys())}")
                    else:
                        logger.info(f"Successfully completed {operation_name} with result: {result}")
                else:
                    logger.warning(f"Operation {operation_name} completed but returned no result")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator

class Message(TypedDict):
    role: str  # "host" or username from livestream
    content: str
    timestamp: datetime
    interaction_type: str  # "local_agent" or "livestream"
    session_id: str

class Session(TypedDict):
    session_id: str
    messages: List[Message]
    created_at: datetime
    last_updated: datetime
    metadata: Optional[dict]

class ContextConfiguration(TypedDict):
    session_id: str
    latest_summary: Optional[dict]  # The most recent summary message
    active_message_ids: List[str]   # IDs of messages in current context
    last_updated: datetime

class WorkflowOperation(TypedDict):
    """Top-level workflow operation"""
    workflow_id: str
    session_id: str
    status: str  # Maps to OperationStatus
    created_at: datetime
    last_updated: datetime
    tool_sequence: List[str]  # List of tool_operation_ids in order
    dependencies: Dict[str, List[str]]  # tool_id -> [dependent_tool_ids]
    metadata: Dict[str, Any]

class ScheduledOperation(TypedDict):
    """Base class for any scheduled operation or content"""
    session_id: str
    workflow_id: Optional[str]  # Link to parent workflow if part of one
    tool_operation_id: Optional[str]  # Link to tool operation that created this
    content_type: str  # Maps to ContentType
    status: str  # Maps to OperationStatus
    count: int  # Number of items to schedule (tweets, posts, etc)
    schedule_type: Literal["immediate", "one_time", "multiple", "recurring"]  # Type of schedule
    schedule_time: str  # When to execute
    approval_required: bool  # Whether approval is needed
    content: Dict[str, Any]  # The actual content to be executed
    pending_items: List[str]  # IDs of items pending approval
    approved_items: List[str]  # IDs of approved items
    rejected_items: List[str]  # IDs of rejected items
    created_at: datetime
    scheduled_time: Optional[datetime]
    executed_time: Optional[datetime]
    metadata: Dict[str, Any]
    retry_count: int
    last_error: Optional[str]
    schedule_id: str
    schedule_state: str  # ScheduleState value
    schedule_info: Dict
    state_history: List[Dict[str, Union[str, datetime]]]  # List of state changes

class ToolItemContent(TypedDict):
    """Base content structure for all tools"""
    raw_content: str
    formatted_content: Optional[str]
    references: Optional[List[str]]
    version: str

class ToolItemParams(TypedDict):
    """Base parameters for all tools"""
    schedule_time: Optional[datetime]
    retry_policy: Optional[Dict]
    execution_window: Optional[Dict]
    custom_params: Dict[str, Any]

class ToolItemMetadata(TypedDict):
    """Base metadata for all tools"""
    generated_at: str
    generated_by: str
    last_modified: str
    version: str

class ToolItemResponse(TypedDict):
    """Base API response for all tools"""
    success: bool
    timestamp: str
    platform_id: Optional[str]
    error: Optional[str]

class OperationMetadata(TypedDict):
    """Standard metadata for operations"""
    content_type: str
    original_request: Optional[str]
    generated_at: str
    execution_time: Optional[str]
    retry_count: int
    last_error: Optional[str]

class ToolExecution(TypedDict):
    """Individual tool execution record"""
    tool_operation_id: str           # Reference to tool_operations
    session_id: str
    tool_type: str             # Maps to ToolType
    state: str                 # Maps to ToolOperationState
    parameters: Dict[str, Any] # Tool-specific parameters
    result: Optional[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime
    metadata: OperationMetadata
    retry_count: int
    last_error: Optional[str]

class ToolItem(TypedDict):
    """Generic tool item"""
    session_id: str
    workflow_id: Optional[str]
    tool_operation_id: str
    content_type: str
    state: str
    status: str
    content: ToolItemContent
    parameters: ToolItemParams
    metadata: ToolItemMetadata
    api_response: Optional[ToolItemResponse]
    created_at: datetime
    scheduled_time: Optional[datetime]
    executed_time: Optional[datetime]
    posted_time: Optional[datetime]
    schedule_id: str
    execution_id: Optional[str]
    retry_count: int
    last_error: Optional[str]

class ToolOperation(TypedDict):
    """Individual tool operation with workflow support"""
    session_id: str
    tool_type: str              # Maps to ToolType enum
    state: str                  # Maps to ToolOperationState
    step: str
    workflow_id: Optional[str]  # Link to parent workflow
    workflow_step: Optional[int] # Order in workflow sequence
    input_data: Dict[str, Any]  # Data from previous tools
    output_data: Dict[str, Any] # Data produced by this tool
    metadata: OperationMetadata # Standard metadata including content_type
    created_at: datetime
    last_updated: datetime
    end_reason: Optional[str]

class TwitterContent(ToolItemContent):
    """Twitter-specific content structure"""
    thread_structure: Optional[List[str]]
    mentions: Optional[List[str]]
    hashtags: Optional[List[str]]
    urls: Optional[List[str]]

class TwitterParams(ToolItemParams):
    """Twitter-specific parameters"""
    custom_params: Dict[str, Any] = {
        "account_id": None,  # Optional[str]
        "media_files": None,  # Optional[List[str]]
        "poll_options": None,  # Optional[List[str]]
        "poll_duration": None,  # Optional[int]
        "reply_settings": None,  # Optional[str]
        "quote_tweet_id": None,  # Optional[str]
        "thread_structure": None,  # Optional[List[str]]
        "mentions": None,  # Optional[List[str]]
        "hashtags": None,  # Optional[List[str]]
        "urls": None,  # Optional[List[str]]
        "audience_targeting": None,  # Optional[Dict]
        "content_category": None,  # Optional[str]
        "sensitivity_level": None,  # Optional[str]
        "estimated_engagement": None,  # Optional[str]
        "visibility_settings": None  # Optional[Dict]
    }

class CalendarParams(ToolItemParams):
    """Calendar-specific parameters"""
    custom_params: Dict[str, Any] = {
        "event_duration": None,  # Optional[int]
        "attendees": None,  # Optional[List[str]]
        "location": None,  # Optional[str]
        "reminder_minutes": None,  # Optional[int]
        "calendar_id": None  # Optional[str]
    }

class TwitterMetadata(ToolItemMetadata):
    """Twitter-specific metadata"""
    estimated_engagement: str
    audience_targeting: Optional[Dict]
    content_category: Optional[str]
    sensitivity_level: Optional[str]

class TwitterResponse(ToolItemResponse):
    """Twitter-specific API response"""
    tweet_id: str
    engagement_metrics: Optional[Dict]
    visibility_stats: Optional[Dict]

class Tweet(ToolItem):
    """Tweet implementation of ToolItem"""
    content: TwitterContent
    parameters: TwitterParams
    metadata: TwitterMetadata
    api_response: Optional[TwitterResponse]

class TwitterCommandAnalysis(BaseModel):
    tools_needed: List[Dict[str, Any]]
    reasoning: str

class TweetGenerationResponse(BaseModel):
    items: List[Dict[str, Any]]

class LimitOrderParams(ToolItemParams):
    """Limit order-specific parameters"""
    custom_params: Dict[str, Any] = {
        # Price Oracle Check (CoinGecko)
        "price_oracle": {
            "symbol": "",  # str
            "target_price_usd": 0.0,  # float
            "last_check": {
                "price_usd": None,  # Optional[float]
                "timestamp": 0  # int
            },
            "check_interval_seconds": 0  # int
        },

        # Add chain information
        "chain_info": {
            "to_chain": "",  # str
            "destination_chain": "",  # str (if different from to_chain)
            "destination_address": ""  # str (if withdrawal enabled)
        },

        # Step 1: Deposit Check & Parameters
        "deposit": {
            "needs_deposit": False,  # bool
            "token_symbol": "",  # str
            "amount": 0.0,  # float
            "requires_wrap": False,  # bool
            "executed": False  # bool
        },

        # Step 2: Swap Parameters
        "swap": {
            "from_token": "",  # str
            "from_amount": 0.0,  # float
            "to_token": "",  # str
            "chain_out": "",  # str
            "executed": False,  # bool
            "current_quote": {  # Optional[Dict]
                "defuse_asset_identifier_in": "",  # str
                "defuse_asset_identifier_out": "",  # str
                "amount_in": "",  # str
                "amount_out": "",  # str
                "expiration_time": "",  # str
                "quote_hash": ""  # str
            }
        },

        # Step 3: Withdrawal Parameters
        "withdraw": {
            "enabled": False,  # bool
            "token_symbol": "",  # str
            "amount": None,  # Optional[float]
            "destination_address": "",  # str
            "destination_chain": "",  # str
            "source_chain": "",  # str
            "executed": False  # bool
        },

        # Execution Control
        "execution": {
            "current_step": "",  # str
            "expiration_timestamp": 0,  # int
            "max_retries": 3,  # int
            "retry_count": 0,  # int
            "last_error": None,  # Optional[str]
            "completed": False  # bool
        }
    }

class RinDB:
    def __init__(self, client: AsyncIOMotorClient):
        self.client = client
        self.db = client['rin_multimodal']
        # Legacy collections for migration
        self.tweets = self.db['rin.tweets']
        self.tweet_schedules = self.db['rin.tweet_schedules']
        
        # Current collections
        self.messages = self.db['rin.messages']
        self.context_configs = self.db['rin.context_configs']
        self.tool_items = self.db['rin.tool_items']
        self.tool_operations = self.db['rin.tool_operations']
        self.tool_executions = self.db['rin.tool_executions']
        self.scheduled_operations = self.db['rin.scheduled_operations']
        self._initialized = False
        logger.info(f"Connected to database: {self.db.name}")

    async def initialize(self):
        """Initialize database and collections"""
        try:
            collections = await self.db.list_collection_names()
            
            # Create collections if they don't exist
            required_collections = [
                'rin.messages',
                'rin.context_configs',
                'rin.tool_items',
                'rin.tool_operations',
                'rin.tool_executions',
                'rin.scheduled_operations'
            ]
            
            for collection in required_collections:
                if collection not in collections:
                    await self.db.create_collection(collection)
                    logger.info(f"Created {collection} collection")
            
            # Setup indexes
            await self._setup_indexes()
            self._initialized = True
            
            # Add index for scheduled operations
            await self.scheduled_operations.create_index([
                ("schedule_state", 1),
                ("content_type", 1)
            ])
            await self.scheduled_operations.create_index([
                ("tool_operation_id", 1)
            ], unique=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def is_initialized(self) -> bool:
        """Check if database is properly initialized"""
        try:
            collections = await self.db.list_collection_names()
            required_collections = [
                'rin.messages',
                'rin.context_configs',
                'rin.tool_items',
                'rin.tool_operations',
                'rin.tool_executions',
                'rin.scheduled_operations'
            ]
            
            has_collections = all(col in collections for col in required_collections)
            
            if not has_collections:
                logger.warning("Required collections not found")
                return False
                
            await self.db.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    async def _setup_indexes(self):
        """Setup indexes for Rin collections"""
        try:
            # Message and context indexes
            await self.messages.create_index([("session_id", 1)])
            await self.messages.create_index([("timestamp", 1)])
            await self.context_configs.create_index([("session_id", 1)])

            # Tool operations indexes
            await self.tool_operations.create_index([("session_id", 1)])
            await self.tool_operations.create_index([("state", 1)])
            await self.tool_operations.create_index([("tool_type", 1)])
            await self.tool_operations.create_index([("last_updated", 1)])
            
            # Tool executions tracking
            await self.tool_executions.create_index([("tool_operation_id", 1)])
            await self.tool_executions.create_index([("session_id", 1)])
            await self.tool_executions.create_index([("state", 1)])
            await self.tool_executions.create_index([("created_at", 1)])
            
            # Tool items (content)
            await self.tool_items.create_index([("session_id", 1)])
            await self.tool_items.create_index([("content_type", 1)])
            await self.tool_items.create_index([("status", 1)])
            await self.tool_items.create_index([("state", 1)])
            await self.tool_items.create_index([("schedule_id", 1)])
            await self.tool_items.create_index([("tool_operation_id", 1)])

            # Temporal indexes
            await self.tool_items.create_index([("created_at", 1)])
            await self.tool_items.create_index([("scheduled_time", 1)])
            await self.tool_items.create_index([("posted_time", 1)])

            # Compound indexes for common queries
            await self.tool_items.create_index([
                ("tool_operation_id", 1),
                ("state", 1)
            ])
            await self.tool_items.create_index([
                ("tool_operation_id", 1),
                ("status", 1)
            ])
            
            # Scheduled operations indexes
            await self.scheduled_operations.create_index([("session_id", 1)])
            await self.scheduled_operations.create_index([("status", 1)])
            await self.scheduled_operations.create_index([("scheduled_time", 1)])
            await self.scheduled_operations.create_index([("content_type", 1)])
            await self.scheduled_operations.create_index([
                ("status", 1),
                ("scheduled_time", 1)
            ])

            logger.info("Successfully created database indexes")
        except Exception as e:
            logger.error(f"Error setting up indexes: {str(e)}")
            raise

    async def add_message(self, session_id: str, role: str, content: str, 
                         interaction_type: str = 'local_agent', metadata: Optional[dict] = None):
        """Add a message to the database
        
        Args:
            session_id: Unique session identifier
            role: Either "host" for local input or username for livestream
            content: Message content
            interaction_type: Either "local_agent" or "livestream"
            metadata: Optional additional metadata
        """
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "interaction_type": interaction_type
        }
        if metadata:
            message["metadata"] = metadata
            
        await self.messages.insert_one(message)
        return message

    async def get_session_messages(self, session_id: str):
        cursor = self.messages.find({"session_id": session_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def clear_session(self, session_id: str):
        await self.messages.delete_many({"session_id": session_id})


    async def update_session_metadata(self, session_id: str, metadata: dict):
        """Update or create session metadata"""
        try:
            await self.messages.update_many(
                {"session_id": session_id},
                {"$set": {"metadata": metadata}},
                upsert=True
            )
            logger.info(f"Updated metadata for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")
            return False

    async def add_context_summary(self, session_id: str, summary: dict, active_message_ids: List[str]):
        """Update context configuration with new summary"""
        await self.context_configs.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "latest_summary": summary,
                    "active_message_ids": active_message_ids,
                    "last_updated": datetime.utcnow()
                }
            },
            upsert=True
        )

    async def get_context_configuration(self, session_id: str) -> Optional[ContextConfiguration]:
        """Get current context configuration"""
        return await self.context_configs.find_one({"session_id": session_id})

    async def get_messages_by_ids(self, session_id: str, message_ids: List[str]) -> List[Message]:
        """Get specific messages by their IDs"""
        cursor = self.messages.find({
            "session_id": session_id,
            "_id": {"$in": [ObjectId(id) for id in message_ids]}
        }).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def create_tool_item(
        self,
        session_id: str,
        content_type: str,
        content: Dict,
        parameters: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new tool item with validation"""
        try:
            # Validate required content
            if not content.get('raw_content'):
                raise ValueError("Tool item content cannot be empty")

            tool_item = {
                "session_id": session_id,
                "content_type": content_type,
                "status": OperationStatus.PENDING.value,
                "content": {
                    **content,
                    "version": "1.0"
                },
                "parameters": {
                    **parameters,
                    "retry_policy": parameters.get("retry_policy", {"max_attempts": 3, "delay": 300})
                },
                "metadata": {
                    **(metadata or {}),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "generated_by": "system",
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0"
                },
                "created_at": datetime.now(timezone.utc),
                "retry_count": 0
            }

            result = await self.tool_items.insert_one(tool_item)
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error creating tool item: {e}")
            raise

    async def get_pending_items(self, 
                              content_type: Optional[str] = None,
                              schedule_id: Optional[str] = None) -> List[Dict]:
        """Get pending items, optionally filtered by type and schedule"""
        try:
            query = {"status": OperationStatus.PENDING.value}
            if content_type:
                query["content_type"] = content_type
            if schedule_id:
                query["schedule_id"] = schedule_id
            
            cursor = self.tool_items.find(query)
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Error fetching pending items: {e}")
            return []

    @db_operation_logger("update_tool_item")
    async def update_tool_item(self, item_id: str, update_data: Dict, return_document: bool = True) -> Optional[Dict]:
        """Update a tool item with enhanced logging and validation"""
        # Validate update_data structure against ToolItem TypedDict
        if not isinstance(update_data, dict):
            raise ValueError("update_data must be a dictionary")
            
        result = await self.tool_items.find_one_and_update(
            {"_id": ObjectId(item_id)},
            {"$set": {
                **update_data,
                "last_updated": datetime.now(timezone.utc)
            }},
            return_document=return_document
        )
        
        # Track the update in tool_executions
        if result:
            await self.tool_executions.insert_one({
                "tool_operation_id": result.get("tool_operation_id"),
                "execution_type": "update",
                "item_id": item_id,
                "updated_fields": list(update_data.keys()),
                "timestamp": datetime.now(timezone.utc)
            })
            
        return result

    @db_operation_logger("update_operation")
    async def update_operation(self, operation_id: str, update_data: Dict, return_document: bool = True) -> Optional[Dict]:
        """Update a tool operation with enhanced logging and validation"""
        # Validate update_data structure against ToolOperation TypedDict
        if not isinstance(update_data, dict):
            raise ValueError("update_data must be a dictionary")
            
        result = await self.tool_operations.find_one_and_update(
            {"_id": ObjectId(operation_id)},
            {"$set": {
                **update_data,
                "last_updated": datetime.now(timezone.utc)
            }},
            return_document=return_document
        )
        
        # Track the update
        if result:
            await self.tool_executions.insert_one({
                "tool_operation_id": operation_id,
                "execution_type": "operation_update",
                "updated_fields": list(update_data.keys()),
                "timestamp": datetime.now(timezone.utc)
            })
            
        return result

    @db_operation_logger("update_schedule")
    async def update_schedule(self, schedule_id: str, update_data: Dict, return_document: bool = True) -> Optional[Dict]:
        """Update a scheduled operation with enhanced logging and validation"""
        # Validate update_data structure against ScheduledOperation TypedDict
        if not isinstance(update_data, dict):
            raise ValueError("update_data must be a dictionary")
            
        result = await self.scheduled_operations.find_one_and_update(
            {"_id": ObjectId(schedule_id)},
            {"$set": {
                **update_data,
                "last_updated": datetime.now(timezone.utc)
            }},
            return_document=return_document
        )
        
        # Track the update
        if result:
            await self.tool_executions.insert_one({
                "tool_operation_id": result.get("tool_operation_id"),
                "execution_type": "schedule_update",
                "schedule_id": schedule_id,
                "updated_fields": list(update_data.keys()),
                "timestamp": datetime.now(timezone.utc)
            })
            
        return result

    @db_operation_logger("update_tool_item_status")
    async def update_tool_item_status(self, 
                                    item_id: str, 
                                    status: OperationStatus,
                                    api_response: Optional[Dict] = None,
                                    error: Optional[str] = None,
                                    metadata: Optional[Dict] = None) -> bool:
        """Update tool item status with enhanced logging and tracking"""
        update_data = {
            "status": status.value if isinstance(status, OperationStatus) else status,
            "last_updated": datetime.now(timezone.utc)
        }
        
        if status == OperationStatus.EXECUTED and api_response:
            update_data["executed_time"] = datetime.now(timezone.utc)
            update_data["api_response"] = api_response
        
        if error:
            update_data["last_error"] = error
            update_data["retry_count"] = 1
        
        if metadata:
            update_data["metadata"] = {
                "$set": {
                    **metadata,
                    "last_modified": datetime.now(timezone.utc).isoformat()
                }
            }
        
        result = await self.tool_items.update_one(
            {"_id": ObjectId(item_id)},
            {"$set": update_data}
        )
        
        # Track the status update
        if result.modified_count > 0:
            await self.tool_executions.insert_one({
                "execution_type": "status_update",
                "item_id": item_id,
                "old_status": None,  # Could fetch this first if needed
                "new_status": status.value if isinstance(status, OperationStatus) else status,
                "has_error": bool(error),
                "timestamp": datetime.now(timezone.utc)
            })
            
        return result.modified_count > 0

    async def set_tool_operation_state(self, session_id: str, operation_data: Dict) -> Optional[Dict]:
        """Set tool operation state"""
        try:
            # Ensure required fields
            operation_data.update({
                "last_updated": datetime.now(timezone.utc)
            })
            
            # Handle both new operations and updates
            result = await self.tool_operations.find_one_and_update(
                {"session_id": session_id},
                {"$set": operation_data},
                upsert=True,
                return_document=True
            )
            
            if result:
                logger.info(f"Set operation state for session {session_id}")
                return result
            else:
                logger.error(f"Failed to set operation state for session {session_id}")
                return None

        except Exception as e:
            logger.error(f"Error setting operation state: {e}")
            return None

    async def get_tool_operation_state(self, session_id: str) -> Optional[Dict]:
        """Get tool operation state"""
        try:
            return await self.tool_operations.find_one({"session_id": session_id})
        except Exception as e:
            logger.error(f"Error getting operation state: {e}")
            return None

    async def get_scheduled_operation(
        self, 
        tool_operation_id: Optional[str] = None,
        status: Optional[str] = None,
        state: Optional[str] = None
    ) -> Optional[Dict]:
        """Get scheduled operation by ID, status, or state"""
        try:
            query = {}
            
            # Build query based on provided parameters
            if tool_operation_id:
                if ObjectId.is_valid(tool_operation_id):
                    query["_id"] = ObjectId(tool_operation_id)
                else:
                    query["$or"] = [
                        {"session_id": tool_operation_id},
                        {"tool_operation_id": tool_operation_id}
                    ]
            
            if status:
                query["status"] = status
                
            if state:
                query["state"] = state
                
            # Execute query
            schedule = await self.scheduled_operations.find_one(query)
            
            if schedule:
                if tool_operation_id:
                    if ObjectId.is_valid(tool_operation_id) and str(schedule['_id']) == tool_operation_id:
                        logger.info(f"Found schedule by ObjectId: {tool_operation_id}")
                    else:
                        logger.info(f"Found schedule by session/operation ID: {tool_operation_id}")
                return schedule
            
            logger.warning(f"No schedule found for query: {query}")
            return None

        except Exception as e:
            logger.error(f"Error getting scheduled operation: {e}")
            return None

    async def create_scheduled_operation(
        self,
        tool_operation_id: str,
        content_type: str,
        schedule_info: Dict,
    ) -> str:
        """Create new scheduled operation"""
        operation = ScheduledOperation(
            schedule_id=str(ObjectId()),
            tool_operation_id=tool_operation_id,
            content_type=content_type,
            schedule_state=ScheduleState.PENDING.value,
            schedule_info=schedule_info,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            state_history=[{
                "state": ScheduleState.PENDING.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "Schedule initialized"
            }],
            metadata={}
        )
        result = await self.scheduled_operations.insert_one(operation)
        return str(result.inserted_id)

    async def update_schedule_state(
        self,
        schedule_id: str,
        state: ScheduleState,
        reason: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update schedule state with history tracking"""
        try:
            # Prepare update operations separately
            set_data = {
                "schedule_state": state.value,
                "last_updated": datetime.now(timezone.utc)
            }
            
            if metadata:
                set_data["metadata"] = metadata

            # Create the update operation
            update_ops = {
                "$set": set_data,
                "$push": {
                    "state_history": {
                        "state": state.value,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "reason": reason
                    }
                }
            }

            result = await self.scheduled_operations.update_one(
                {"_id": ObjectId(schedule_id)},
                update_ops
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated schedule {schedule_id} state to {state.value}")
            else:
                logger.warning(f"No schedule updated for ID: {schedule_id}")
                
            return success

        except Exception as e:
            logger.error(f"Error updating schedule state: {e}")
            return False

    async def delete_all_scheduled_tweets(self):
        """Delete all tweet schedules and their associated tweets"""
        try:
            # Get all schedule IDs first
            schedule_cursor = self.scheduled_operations.find({})
            schedules = await schedule_cursor.to_list(length=None)
            schedule_ids = [str(schedule['_id']) for schedule in schedules]
            
            # Delete all tweets associated with these schedules
            for schedule_id in schedule_ids:
                await self.tool_items.delete_many({"schedule_id": schedule_id})
                logger.info(f"Deleted items for schedule {schedule_id}")
            
            # Delete all tweet schedules
            result = await self.scheduled_operations.delete_many({})
            
            logger.info(f"Deleted {result.deleted_count} scheduled operations")
            return {
                "operations_deleted": result.deleted_count,
                "schedule_ids": schedule_ids
            }
            
        except Exception as e:
            logger.error(f"Error deleting scheduled operations: {e}")
            raise

    async def update_scheduled_operation(
        self,
        schedule_id: str,
        state: Optional[str] = None,
        schedule_state: Optional[str] = None,
        status: Optional[str] = None,
        pending_item_ids: Optional[List[str]] = None,
        approved_item_ids: Optional[List[str]] = None,
        rejected_item_ids: Optional[List[str]] = None,
        schedule_info: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update a scheduled operation"""
        try:
            update_data = {}
            if state is not None:
                update_data["state"] = state
            if schedule_state is not None:
                update_data["schedule_state"] = schedule_state
            if status is not None:
                update_data["status"] = status
            if pending_item_ids is not None:
                update_data["pending_items"] = pending_item_ids
            if approved_item_ids is not None:
                update_data["approved_items"] = approved_item_ids
            if rejected_item_ids is not None:
                update_data["rejected_items"] = rejected_item_ids
            if schedule_info is not None:
                update_data["schedule_info"] = schedule_info
            if metadata is not None:
                if "state_history" in metadata:
                    update_data["state_history"] = metadata.pop("state_history")
                update_data["metadata"] = {
                    **update_data.get("metadata", {}),
                    **metadata,
                    "last_modified": datetime.now(timezone.utc).isoformat()
                }

            final_update = {"$set": update_data}
            result = await self.scheduled_operations.update_one(
                {"_id": ObjectId(schedule_id) if ObjectId.is_valid(schedule_id) else schedule_id},
                final_update
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated scheduled operation: {schedule_id}")
            else:
                logger.warning(f"No scheduled operation updated for ID: {schedule_id}")
                
            return success

        except Exception as e:
            logger.error(f"Error updating schedule state: {e}", exc_info=True)
            return False

    @db_operation_logger("store_tool_item_content")
    async def store_tool_item_content(
        self,
        item_id: str,
        content: Dict,
        operation_details: Dict,
        source: str,  # 'analyze_command' or 'generate_content'
        tool_operation_id: str
    ) -> Optional[Dict]:
        """Store and validate tool item content with detailed logging"""
        try:
            logger.info(f"Storing content from {source} for item {item_id}")
            logger.info(f"Operation details: {json.dumps(operation_details, indent=2)}")
            logger.info(f"Content structure: {json.dumps(content, indent=2)}")

            # Validate required fields based on source
            if source == 'analyze_command':
                required_fields = ['from_token', 'from_amount', 'to_token', 'target_price_usd', 'reference_token']
                missing_fields = [f for f in required_fields if f not in operation_details]
                if missing_fields:
                    logger.error(f"Missing required fields from analyze_command: {missing_fields}")
                    return None

            # Update the item with content validation
            update_data = {
                "content": content,
                "operation_details": operation_details,
                "metadata": {
                    "content_source": source,
                    "content_updated_at": datetime.now(timezone.utc).isoformat(),
                    "content_validation": {
                        "has_operation_details": bool(operation_details),
                        "has_content": bool(content),
                        "source": source
                    }
                }
            }

            # For regeneration, add additional tracking but don't overwrite existing fields
            if source == 'analyze_command_regeneration':
                update_data["metadata"].update({
                    "regeneration_analysis": {
                        "analyzed_params": operation_details,
                        "analyzed_at": datetime.now(timezone.utc).isoformat()
                    }
                })

            result = await self.tool_items.find_one_and_update(
                {"_id": ObjectId(item_id)},
                {"$set": update_data},
                return_document=True
            )

            if result:
                # Track content update in tool_executions
                await self.tool_executions.insert_one({
                    "tool_operation_id": tool_operation_id,
                    "execution_type": "content_update",
                    "item_id": item_id,
                    "source": source,
                    "content_fields": list(content.keys()),
                    "operation_detail_fields": list(operation_details.keys()),
                    "timestamp": datetime.now(timezone.utc)
                })

                logger.info(f"Successfully stored content for item {item_id}")
                logger.info(f"Stored content fields: {list(content.keys())}")
                logger.info(f"Stored operation details: {list(operation_details.keys())}")
                return result
            else:
                logger.error(f"Failed to store content for item {item_id}")
                return None

        except Exception as e:
            logger.error(f"Error storing tool item content: {e}", exc_info=True)
            raise