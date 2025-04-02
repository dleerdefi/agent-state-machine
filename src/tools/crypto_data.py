# must be updated to use new tool structure
from datetime import datetime, timezone
import logging
import asyncio
from typing import Dict, Optional, Any, List
import json
from bson import ObjectId

from src.tools.base import (
    BaseTool,
    AgentResult,
    AgentDependencies,
    ToolRegistry
)
from src.clients.coingecko_client import CoinGeckoClient
from src.db.enums import OperationStatus, ToolOperationState, ContentType, ToolType
from src.utils.json_parser import parse_strict_json

logger = logging.getLogger(__name__)

class CryptoTool(BaseTool):
    """Cryptocurrency price and market data tool"""
    
    # Static tool configuration
    name = "crypto_data"  # Match the ToolType.CRYPTO_DATA value exactly
    description = "Get cryptocurrency price and market data"
    version = "1.0.0"
    
    # Tool registry configuration - optimized for one-shot usage
    registry = ToolRegistry(
        content_type=ContentType.CRYPTO_DATA,
        tool_type=ToolType.CRYPTO_DATA,
        requires_approval=False,  # No approval needed
        requires_scheduling=False,  # No scheduling needed
        required_clients=["coingecko_client"],
        required_managers=["tool_state_manager"]
    )
    
    def __init__(self, deps: Optional[AgentDependencies] = None):
        """Initialize crypto tool with dependencies"""
        super().__init__()
        self.deps = deps or AgentDependencies()
        
        # Services will be injected by orchestrator based on registry requirements
        self.tool_state_manager = None
        self.coingecko_client = None
        self.db = None

    def inject_dependencies(self, **services):
        """Inject required services - called by orchestrator during registration"""
        self.tool_state_manager = services.get("tool_state_manager")
        self.coingecko_client = services.get("coingecko_client")
        self.db = self.tool_state_manager.db if self.tool_state_manager else None
    
    async def run(self, input_data: str) -> Dict:
        """Run the crypto tool - optimized for one-shot use without approval flow"""
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
            
            # Analyze command to extract token symbol
            command_analysis = await self._analyze_command(input_data)
            logger.info(f"Crypto tool analysis: {command_analysis}")
            
            # Generate content (fetch data)
            content_result = await self._generate_content(
                symbol=command_analysis.get("symbol", "BTC"),
                include_details=command_analysis.get("include_details", True),
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
            formatted_response = content_result.get("response", "No cryptocurrency data available.")
            
            return {
                "status": "completed",
                "state": ToolOperationState.COMPLETED.value,
                "response": formatted_response,
                "requires_chat_response": True,
                "data": content_result.get("data", {})
            }
            
        except Exception as e:
            logger.error(f"Error in crypto tool: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "response": f"Sorry, I couldn't retrieve cryptocurrency data: {str(e)}",
                "requires_chat_response": True
            }

    async def _analyze_command(self, command: str) -> Dict:
        """Analyze command to extract token symbol and parameters"""
        try:
            # Extract token symbol from command
            symbol = "BTC"  # Default to Bitcoin
            include_details = True
            
            # Simple keyword matching for token symbols
            common_tokens = {
                "btc": "BTC", "bitcoin": "BTC",
                "eth": "ETH", "ethereum": "ETH",
                "sol": "SOL", "solana": "SOL",
                "near": "NEAR",
                "usdc": "USDC", "usdt": "USDT",
                "doge": "DOGE", "dogecoin": "DOGE",
                "xrp": "XRP", "ripple": "XRP",
                "ada": "ADA", "cardano": "ADA"
            }
            
            # Check for token names in command
            words = command.lower().split()
            for word in words:
                if word in common_tokens:
                    symbol = common_tokens[word]
                    break
            
            # Check for more detailed analysis request
            include_details = any(word in command.lower() for word in 
                              ["detail", "market", "info", "data", "volume", "cap"])
            
            return {
                "symbol": symbol,
                "include_details": include_details,
                "item_count": 1  # Always 1 for this tool
            }
            
        except Exception as e:
            logger.error(f"Error analyzing crypto command: {e}")
            return {"symbol": "BTC", "include_details": True, "item_count": 1}

    async def _generate_content(
        self, 
        symbol: str = None,
        include_details: bool = True,
        tool_operation_id: Optional[str] = None,
        # Add parameters to match orchestrator's calling convention
        topic: Optional[str] = None,
        count: int = 1,
        analyzed_params: Optional[Dict] = None
    ) -> Dict:
        """Generate crypto data content. Returns dict with data and user response.
           Does NOT interact with the database directly.
        """
        # Check if client is available
        if not self.coingecko_client:
            logger.error("CoinGecko client is not initialized or injected correctly.")
            return {
                "status": "error",
                "error": "Crypto client not available",
                "content_to_store": None,
                "response": "Sorry, the crypto client isn't available."
            }
            
        try:
            # Use symbol from parameters or extract from topic/analyzed_params if not provided
            if not symbol:
                if analyzed_params and "symbol" in analyzed_params:
                    symbol = analyzed_params["symbol"]
                elif topic:
                    # Try to extract symbol from topic
                    for common_token in ["BTC", "ETH", "SOL", "NEAR", "USDC", "USDT", "DOGE", "XRP", "ADA"]:
                        if common_token in topic.upper():
                            symbol = common_token
                            break
                
            if not symbol:
                symbol = "BTC"  # Default to Bitcoin if no symbol is found
                
            logger.info(f"Fetching crypto data for symbol: {symbol}")
                
            coingecko_id = await self.coingecko_client._get_coingecko_id(symbol)
            if not coingecko_id:
                raise ValueError(f"Could not find CoinGecko ID for symbol: {symbol}")
                
            tasks = [self.coingecko_client.get_token_price(coingecko_id)]
            if include_details:
                tasks.append(self.coingecko_client.get_token_details(coingecko_id))
            results = await asyncio.gather(*tasks)
            
            data = {}
            for result in results:
                if result: data.update(result)
            
            content_to_store = {
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            formatted_response = self._format_crypto_response(data)
            
            return {
                "status": "success",
                "data": data,
                "content_to_store": content_to_store,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": formatted_response
            }

        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return {
                "status": "error", "error": str(e),
                "content_to_store": None,
                "response": f"Sorry, error fetching crypto data for {symbol}: {e}"
            }

    def _format_crypto_response(self, data: Dict) -> str:
        """Format cryptocurrency data for user consumption"""
        try:
            response_parts = []
            
            # Basic price info
            if 'price_usd' in data:
                response_parts.append(f"💰 Current Price: ${data['price_usd']:,.2f} USD")
            
            # Price changes
            changes = {
                '24h': data.get('price_change_24h'),
                '7d': data.get('price_change_7d'),
                '30d': data.get('price_change_30d')
            }
            change_strs = []
            for period, change in changes.items():
                if change is not None:
                    emoji = "📈" if change > 0 else "📉"
                    change_strs.append(f"{emoji} {period}: {change:+.2f}%")
            if change_strs:
                response_parts.append("Price Changes:")
                response_parts.extend(change_strs)
            
            # Market data
            if 'market_cap' in data:
                response_parts.append(f"🌐 Market Cap: ${data['market_cap']:,.0f}")
            if 'total_volume' in data:
                response_parts.append(f"📊 24h Volume: ${data['total_volume']:,.0f}")
            
            # Supply info
            if any(k in data for k in ['circulating_supply', 'total_supply', 'max_supply']):
                response_parts.append("Supply Information:")
                if 'circulating_supply' in data:
                    response_parts.append(f"  • Circulating: {data['circulating_supply']:,.0f}")
                if 'total_supply' in data:
                    response_parts.append(f"  • Total: {data['total_supply']:,.0f}")
                if 'max_supply' in data:
                    response_parts.append(f"  • Max: {data['max_supply']:,.0f}")
            
            # Social metrics (if available)
            social_metrics = {
                'twitter_followers': '𝕏',
                'reddit_subscribers': '📱',
                'telegram_channel_user_count': '📢'
            }
            social_data = []
            for key, emoji in social_metrics.items():
                if data.get(key):
                    social_data.append(f"{emoji} {key.replace('_', ' ').title()}: {data[key]:,}")
            if social_data:
                response_parts.append("\nSocial Metrics:")
                response_parts.extend(social_data)
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error formatting crypto response: {e}")
            return str(data)  # Fallback to basic string representation

    def can_handle(self, command_text: str, tool_type: Optional[str] = None) -> bool:
        """Check if this tool can handle the given command"""
        # If tool_type is explicitly specified as 'crypto_data', handle it
        if tool_type and tool_type.lower() == self.registry.tool_type.value.lower():
            return True
        
        # Otherwise, don't try to detect keywords here - that's the trigger detector's job
        return False