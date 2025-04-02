from enum import Enum

class OperationStatus(str, Enum):
    """Status of any operation in the system"""
    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress" # not used in existing enums.py
    REJECTED = "rejected"
    SCHEDULED = "scheduled"
    EXECUTED = "executed"
    FAILED = "failed"

class ToolOperationState(str, Enum):
    """State of a tool operation"""
    INACTIVE = "inactive"
    COLLECTING = "collecting"
    APPROVING = "approving"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    
class ScheduleState(str, Enum):
    """State of a scheduled operation"""
    PENDING = "pending"
    ACTIVATING = "activating"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

class ApprovalState(str, Enum):
    """State of an approval request"""
    AWAITING_INITIAL = "awaiting_initial"
    AWAITING_APPROVAL = "awaiting_approval"
    REGENERATING = "regenerating"
    APPROVAL_FINISHED = "approval_finished"
    APPROVAL_CANCELLED = "approval_cancelled"

class ContentType(str, Enum):
    """Type of content being processed"""
    TWEET = "tweet"
    THREAD = "thread"
    LIMIT_ORDER = "limit_order"
    CALENDAR_EVENT = "calendar_event"
    REMINDER = "reminder"
    TASK = "task"
    MESSAGE = "message"
    EMAIL = "email"
    CRYPTO_DATA = "crypto_data"
    SEARCH_RESULTS = "search_results"

class ToolType(Enum):
    """Tool types supported by system"""
    TWITTER = "twitter"
    NEAR = "near"
    CALENDAR = "calendar"
    REMINDER = "reminder"
    TASK_MANAGER = "task_manager"
    EMAIL = "email"
    LIMIT_ORDER = "limit_order"
    INTERNAL = "internal"
    INTENTS = "intents"
    TIME = "time"
    WEATHER = "weather"
    CRYPTO_DATA = "crypto_data"  # New type for CryptoTool
    SEARCH = "search"  # New type for PerplexityTool 

class AgentState(str, Enum):
    """State of the agent"""
    NORMAL_CHAT = "normal_chat"
    TOOL_OPERATION = "tool_operation"

class AgentAction(str, Enum): # actions are in the manager scripts
    """Actions that can trigger agent state transitions"""
    START_TOOL = "start_tool"
    COMPLETE_TOOL = "complete_tool"
    CANCEL_TOOL = "cancel_tool"
    ERROR = "error"

class ScheduleAction(str, Enum):
    """Actions that can trigger schedule state transitions"""
    INITIALIZE = "initialize"
    ACTIVATE = "activate"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    ERROR = "error"
    COMPLETE = "complete"