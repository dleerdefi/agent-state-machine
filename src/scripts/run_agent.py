"""
Simple startup script for running the agent with WebSocket interface
"""

import asyncio
import logging
import os
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse
import signal
import threading
from src.utils.logging_config import setup_logging
from dotenv import load_dotenv

# Define project_root at the module level
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load environment variables from .env file
load_dotenv(os.path.join(project_root, '.env'))

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import needed modules
from src.agent.agent import RinAgent
from src.db.mongo_manager import MongoManager
from src.services.websocket_server import ChatWebSocketServer
from src.services.schedule_service import ScheduleService
from src.services.monitoring_service import LimitOrderMonitoringService
from src.clients.coingecko_client import CoinGeckoClient
from src.clients.near_account_helper import get_near_account
from src.db.enums import ToolType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"agent_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    ]
)
logger = logging.getLogger("agent_server")

# Global variables
stop_event = threading.Event()
agent = None
ws_server = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal. Cleaning up...")
    stop_event.set()

def load_config():
    """Load configuration from config.json or environment variables"""
    try:
        config_path = Path(project_root) / 'config' / 'config.json'
        logger.info(f"Attempting to load config from {config_path}")
        
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config_str = f.read()
                
                # Replace environment variables
                for key, value in os.environ.items():
                    config_str = config_str.replace(f"${{{key}}}", value)
                
                config = json.loads(config_str)
                
            # Transform config for agent
            agent_config = {
                'mongo_uri': config.get('mongodb', {}).get('uri'),
                'near_account_id': config.get('near', {}).get('account_id'),
                'near_private_key': config.get('near', {}).get('private_key'),
                'near_network': config.get('near', {}).get('network', 'testnet'),
                'coingecko_api_key': config.get('keys', [{}])[0].get('COINGECKO_API_KEY')
            }
        else:
            # Use environment variables if config.json doesn't exist
            logger.info("Config file not found, using environment variables")
            agent_config = {
                'mongo_uri': os.getenv('MONGO_URI'),
                'near_account_id': os.getenv('NEAR_ACCOUNT_ID'),
                'near_private_key': os.getenv('NEAR_PRIVATE_KEY'),
                'near_network': os.getenv('NEAR_NETWORK', 'testnet'),
                'coingecko_api_key': os.getenv('COINGECKO_API_KEY')
            }
        
        # Ensure we don't return empty values that could be misinterpreted as hostnames
        return {k: v for k, v in agent_config.items() if v}
        
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        # Return a default config or raise the error
        return {}  # or raise

async def handle_websocket_client(websocket, path):
    """Handle WebSocket connection"""
    global agent
    session_id = f"web_{id(websocket)}"
    
    try:
        # Initialize session
        welcome_msg = await agent.start_new_session(session_id)
        await websocket.send(json.dumps({
            "author": "Agent",
            "content": welcome_msg,
            "timestamp": datetime.now().isoformat()
        }))
        
        async for message in websocket:
            if stop_event.is_set():
                break
                
            try:
                # Parse message
                message_data = json.loads(message)
                user_message = message_data.get('content', '')
                
                logger.info(f"Received message: {user_message[:50]}...")
                
                # Process through agent
                response = await agent.get_response(
                    session_id=session_id,
                    message=user_message,
                    role="user",
                    interaction_type="local_agent"
                )
                
                # Send response
                await websocket.send(json.dumps({
                    "author": "Agent",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                }))
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send(json.dumps({
                    "author": "System",
                    "content": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        logger.info(f"Client disconnected: {session_id}")

async def main():
    global agent, ws_server
    
    try:
        # Setup logging
        console = setup_logging()
        
        # Load configuration
        config = load_config()
        
        # Get MongoDB URI directly from environment variable
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            logging.error("MONGO_URI not found in environment variables")
            return
            
        logging.info(f"Connecting to MongoDB using URI from .env file: {mongo_uri[:15]}***")
        
        # Initialize MongoDB with the URI from .env
        await MongoManager.initialize(mongo_uri)
        
        # Get RinDB instance
        db = MongoManager.get_db()
        
        # Verify that the db is a proper RinDB instance with the needed methods
        if not hasattr(db, 'store_tool_item_content'):
            logging.error("ERROR: The database instance doesn't have the required methods")
            logging.error(f"Type of db: {type(db).__name__}")
            logging.error("This will cause errors when processing limit orders")
            logging.error("Attempting to re-initialize the RinDB instance")
            
            # Re-import and try to manually create a RinDB instance
            from src.db.db_schema import RinDB
            db = RinDB(MongoManager._instance)
            await db.initialize()
            
            # Verify again
            if not hasattr(db, 'store_tool_item_content'):
                logging.error("Still failed to get a proper RinDB instance. Agent will have errors.")
        else:
            logging.info("Successfully verified RinDB instance with required methods")
        
        # Ensure config has mongo_uri set
        if not config.get('mongo_uri'):
            config['mongo_uri'] = mongo_uri
        
        # Initialize the agent with db
        agent = RinAgent(config, db)
        await agent.initialize()
        
        # Get NEAR account for limit orders/intents
        near_account = get_near_account()
        if near_account:
            logger.info("NEAR account initialized successfully")
            
            # Inject NEAR account into agent's orchestrator
            if hasattr(agent, 'orchestrator'):
                agent.orchestrator.near_account = near_account
                # Re-register IntentsTool with NEAR account
                agent.orchestrator._register_intents_tool()
                logger.info("NEAR account injected into orchestrator")
                
                # Get IntentsTool from registry
                intents_tool = agent.orchestrator.tools.get(ToolType.INTENTS.value)
                if intents_tool:
                    logger.info("IntentsTool registered successfully")
                else:
                    logger.warning("IntentsTool not found in registry")
        else:
            logger.warning("NEAR account initialization failed - limit orders will not work")
        
        # Initialize CoinGecko client for price monitoring
        coingecko_client = CoinGeckoClient(api_key=config.get('coingecko_api_key'))
        
        # Inject dependencies into monitoring service
        if hasattr(agent, 'monitoring_service'):
            await agent.monitoring_service.inject_dependencies(
                coingecko_client=coingecko_client,
                schedule_manager=agent.orchestrator.schedule_manager,
                near_account=near_account,
                intents_tool=intents_tool if 'intents_tool' in locals() else None
            )
            
            # Ensure monitoring service is set in orchestrator
            agent.orchestrator.set_monitoring_service(agent.monitoring_service)
        
        # Start WebSocket server
        ws_server = ChatWebSocketServer(agent=agent)
        await ws_server.start()
        
        logger.info("Agent server started successfully")
        
        # Keep running until stopped
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        # Clean up connections
        try:
            await MongoManager.close()
        except:
            pass
        raise e
    finally:
        # Cleanup
        logger.info("Shutting down...")
        
        # First stop services
        try:
            if agent:
                await agent.cleanup()
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")
        
        # Then close the websocket server
        try:
            if ws_server:
                await ws_server.stop()
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
        
        logger.info("Shutdown complete")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Ensure clean exit
        logger.info("Exiting...")
        sys.exit(0) 