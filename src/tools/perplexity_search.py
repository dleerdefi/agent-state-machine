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
        # Add parameters to match orchestrator's calling convention
        topic: Optional[str] = None,
        count: int = 1,
        revision_instructions: Optional[str] = None,
        schedule_id: Optional[str] = None,
        analyzed_params: Optional[Dict] = None
    ) -> Dict:
        """Generate search content - compatible with orchestrator's calling convention"""
        try:
            # Use query from parameters or extract from topic/analyzed_params if not provided
            if not query:
                if analyzed_params and "query" in analyzed_params:
                    query = analyzed_params["query"]
                elif topic:
                    query = topic
                    
            if not query:
                raise ValueError("No search query provided")
                
            if not self.perplexity_client:
                raise ValueError("Perplexity client not configured")
                
            # Log the search query
            logger.info(f"Performing search for query: {query}")
                
            # Perform search
            result = await self.perplexity_client.search(query, max_tokens)
            
            # Create a single item for this search result with a proper ObjectId
            item_id = ObjectId()
            
            item = {
                "_id": item_id,
                "content": {
                    "query": query,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "status": OperationStatus.EXECUTED.value,
                "state": ToolOperationState.COMPLETED.value
            }
            
            # Store in database if we have tool_operation_id
            if tool_operation_id and hasattr(self.db, 'store_tool_item_content'):
                await self.db.store_tool_item_content(
                    item_id=str(item_id),  # Convert ObjectId to string
                    content=item.get("content", {}),
                    operation_details={"query": query, "max_tokens": max_tokens},
                    source='generate_content',
                    tool_operation_id=tool_operation_id
                )
            
            # Format response with sources for immediate display
            search_response = result.get("answer", "No results found.")
            sources = result.get("sources", [])
            
            if sources:
                source_text = "\n\nSources:\n" + "\n".join([
                    f"- {s.get('title', 'Untitled')} ({s.get('url', 'No URL')})"
                    for s in sources[:3]  # Limit to first 3 sources
                ])
                search_response += source_text
            
            return {
                "status": "success",
                "data": {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", [])
                },
                "items": [item],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": search_response  # Add response for orchestrator
            }

        except Exception as e:
            logger.error(f"Error in perplexity search: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "items": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def can_handle(self, command_text: str, tool_type: Optional[str] = None) -> bool:
        """Check if this tool can handle the given command"""
        # If tool_type is explicitly specified as 'search', handle it
        if tool_type and tool_type.lower() == self.registry.tool_type.value.lower():
            return True
        
        # Otherwise, don't try to detect keywords here - that's the trigger detector's job
        return False 