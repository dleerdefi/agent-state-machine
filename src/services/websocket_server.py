"""
WebSocket server for agent chat interface
"""

import logging
import asyncio
import json
import websockets
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ChatWebSocketServer:
    def __init__(self, agent, host: str = "0.0.0.0", port: int = 8765):
        self.agent = agent
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
    
    async def handle_client(self, websocket):
        """Handle client connection"""
        try:
            self.clients.add(websocket)
            session_id = f"web_{id(websocket)}"
            
            # Send welcome message
            welcome_msg = await self.agent.start_new_session(session_id)
            await websocket.send(json.dumps({
                "author": "Agent",
                "content": welcome_msg,
                "timestamp": datetime.now().isoformat(),
                "type": "assistant_message"
            }))
            
            async for message in websocket:
                try:
                    # Parse message
                    message_data = json.loads(message)
                    user_message = message_data.get("content", "")
                    author = message_data.get("author", "User")
                    
                    if not user_message.strip():
                        continue
                    
                    # Process message through agent
                    response = await self.agent.get_response(
                        session_id=session_id,
                        message=user_message,
                        role="user",
                        interaction_type="local_agent"
                    )
                    
                    # Send response
                    await websocket.send(json.dumps({
                        "author": "Agent",
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                        "type": "assistant_message"
                    }))
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "author": "System",
                        "content": f"Error: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "system_message" 
                    }))
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up when client disconnects
            if websocket in self.clients:
                self.clients.remove(websocket)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
            
        # Convert to JSON if not already a string
        if not isinstance(message, str):
            message = json.dumps(message)
            
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.add(client)
                
        # Remove disconnected clients
        for client in disconnected:
            self.clients.remove(client)
    
    async def start(self):
        """Start the WebSocket server"""
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            # Create the server but don't block
            self.server = await websockets.serve(self.handle_client, self.host, self.port)
            logger.info(f"WebSocket server running at ws://{self.host}:{self.port}")
            
            # Start HTTP server for static files
            import http.server
            import socketserver
            import threading
            from pathlib import Path
            
            class MyHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    static_dir = Path(__file__).parent.parent / "static"
                    super().__init__(*args, directory=str(static_dir), **kwargs)
            
            http_port = 8766
            http_server = socketserver.TCPServer(("", http_port), MyHandler)
            
            def run_http_server():
                logger.info(f"HTTP server started on http://{self.host}:{http_port}")
                http_server.serve_forever()
            
            thread = threading.Thread(target=run_http_server, daemon=True)
            thread.start()
            
            return self.server
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            logger.info("WebSocket server stopped") 