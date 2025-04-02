# must be updated to use new tool structure
from datetime import datetime, timezone
import logging
from typing import Dict, Optional, Any, List
import json
from bson import ObjectId

from src.tools.base import (
    BaseTool,
    AgentResult,
    AgentDependencies,
    ToolRegistry
)
from src.clients.perplexity_client import PerplexityClient
from src.db.enums import OperationStatus, ToolOperationState, ContentType, ToolType
from src.utils.json_parser import parse_strict_json

logger = logging.getLogger(__name__)

class PerplexityTool(BaseTool):
    """Real-time web search and information retrieval tool"""
    
    # Static tool configuration
    name = "search"
    description = "Real-time web search and information retrieval tool"
    version = "1.0.0"
    
    # Tool registry configuration - optimized for one-shot usage
    registry = ToolRegistry(
        content_type=ContentType.SEARCH_RESULTS,
        tool_type=ToolType.SEARCH,
        requires_approval=False,  # No approval needed
        requires_scheduling=False,  # No scheduling needed
        required_clients=["perplexity_client"],
        required_managers=["tool_state_manager"]
    )
    
    def __init__(self, deps: Optional[AgentDependencies] = None):
        """Initialize perplexity tool with dependencies"""
        super().__init__()
        self.deps = deps or AgentDependencies()
        
        # Services will be injected by orchestrator based on registry requirements
        self.tool_state_manager = None
        self.perplexity_client = None
        self.db = None

    def inject_dependencies(self, **services):
        """Inject required services - called by orchestrator during registration"""
        self.tool_state_manager = services.get("tool_state_manager")
        self.perplexity_client = services.get("perplexity_client")
        self.db = self.tool_state_manager.db if self.tool_state_manager else None

    async def run(self, input_data: str) -> Dict:
        """Run the perplexity tool - optimized for one-shot use without approval flow"""
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
            
            # Analyze command to extract search parameters
            command_analysis = await self._analyze_command(input_data)
            logger.info(f"Perplexity tool analysis: {command_analysis}")
            
            # Generate content (perform search)
            content_result = await self._generate_content(
                query=command_analysis.get("query", input_data),
                max_tokens=command_analysis.get("max_tokens", 300),
                tool_operation_id=str(operation["_id"]),
                topic=input_data,
                count=1,
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
            search_response = content_result.get("response", "No results found.")
            
            return {
                "status": "completed",
                "state": ToolOperationState.COMPLETED.value,
                "response": search_response,
                "requires_chat_response": True,
                "data": content_result.get("data", {})
            }
            
        except Exception as e:
            logger.error(f"Error in perplexity tool: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "response": f"Sorry, I couldn't retrieve search results: {str(e)}",
                "requires_chat_response": True
            }

    async def _analyze_command(self, command: str) -> Dict:
        """Analyze command to extract search parameters"""
        try:
            # Default parameters
            query = command
            max_tokens = 300
            
            # Extract max_tokens if specified (e.g., "detailed search about quantum computing")
            if any(word in command.lower() for word in ["detailed", "comprehensive", "in-depth", "thorough"]):
                max_tokens = 500
            elif any(word in command.lower() for word in ["brief", "short", "summary", "quick"]):
                max_tokens = 200
                
            return {
                "query": query,
                "max_tokens": max_tokens,
                "item_count": 1  # Always 1 for this tool
            }
            
        except Exception as e:
            logger.error(f"Error analyzing search command: {e}")
            return {"query": command, "max_tokens": 300, "item_count": 1}

    async def _generate_content(
        self, 
        query: str = None,
        max_tokens: int = 300,
        tool_operation_id: Optional[str] = None,
        topic: Optional[str] = None,
        count: int = 1,
        revision_instructions: Optional[str] = None,
        schedule_id: Optional[str] = None,
        analyzed_params: Optional[Dict] = None
    ) -> Dict:
        """Generate search content. Aligns with perplexity_client return structure.
           Does NOT interact with the database directly.
        """
        if not self.perplexity_client:
            logger.error("Perplexity client is not initialized or injected correctly.")
            return {
                "status": "error", "error": "Search client not available",
                "content_to_store": None, "response": "Sorry, the search client isn't available.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        try:
            if not query:
                if analyzed_params and "query" in analyzed_params: query = analyzed_params["query"]
                elif topic: query = topic
            if not query: raise ValueError("No search query provided")
                
            logger.info(f"Performing search for query: {query}")
            client_result = await self.perplexity_client.search(query, max_tokens)

            # Check the status returned by the client
            if client_result.get("status") != "success":
                error_msg = client_result.get("error", "Unknown client error")
                logger.error(f"Perplexity client failed: {error_msg}")
                return {
                    "status": "error", "error": error_msg,
                    "content_to_store": None,
                    "response": f"Sorry, the search failed: {error_msg}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Extract the raw answer string from the client's 'data' field
            # The client returns {"status": "success", "data": "answer string"}
            answer_string = client_result.get("data", "")
            if not answer_string:
                 logger.warning(f"Perplexity client returned success but no answer data for query: {query}")
                 answer_string = "No specific answer found."

            # Prepare content for database storage
            content_to_store = {
                "query": query,
                "answer": answer_string, # Store the raw answer string
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # User-facing response is the answer string itself
            search_response = answer_string 
            
            # Create items array for orchestrator expected structure
            items = [{
                "_id": str(ObjectId()),
                "content": content_to_store,
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            }]
            
            return {
                "status": "success",
                "data": {"answer": answer_string},
                "content_to_store": content_to_store, 
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": search_response,
                "items": items
            }

        except Exception as e:
            logger.error(f"Error in perplexity search _generate_content: {e}", exc_info=True)
            return {
                "status": "error", "error": str(e),
                "content_to_store": None,
                "response": f"Sorry, an error occurred during the search: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def can_handle(self, command_text: str, tool_type: Optional[str] = None) -> bool:
        """Check if this tool can handle the given command"""
        # If tool_type is explicitly specified as 'search', handle it
        if tool_type and tool_type.lower() == self.registry.tool_type.value.lower():
            return True
        
        # Otherwise, don't try to detect keywords here - that's the trigger detector's job
        return False 