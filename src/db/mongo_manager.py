from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class MongoManager:
    _instance: Optional[AsyncIOMotorClient] = None
    _db = None

    @classmethod
    async def initialize(cls, mongo_uri: str, max_retries: int = 3):
        """Initialize MongoDB connection with retries"""
        if cls._instance is not None:
            logger.info("MongoManager already initialized - returning existing instance")
            return cls._instance

        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Initializing MongoDB connection (attempt {retry_count + 1})")
                cls._instance = AsyncIOMotorClient(mongo_uri)
                
                # Test connection
                await cls._instance.admin.command('ping')
                
                # Import here to avoid circular import
                from src.db.db_schema import RinDB
                
                # Initialize RinDB with the client
                logger.info("Creating RinDB instance")
                cls._db = RinDB(cls._instance)
                
                # Log the type immediately after creation
                logger.info(f"Created cls._db of type: {type(cls._db).__name__}")
                
                # Check if the instance has expected methods before initializing
                if not hasattr(cls._db, 'initialize'):
                    logger.error("ERROR: RinDB instance doesn't have 'initialize' method!")
                    raise RuntimeError("Invalid RinDB instance created")
                
                if not hasattr(cls._db, 'store_tool_item_content'):
                    logger.error("ERROR: RinDB instance doesn't have 'store_tool_item_content' method!")
                    raise RuntimeError("Invalid RinDB instance created")
                
                await cls._db.initialize()
                
                logger.info("MongoDB connection and collections verified")
                logger.info(f"Final cls._db type: {type(cls._db).__name__}")
                
                return cls._instance
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to initialize MongoDB after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"MongoDB connection attempt {retry_count} failed: {e}. Retrying...")
                await asyncio.sleep(1)

    @classmethod
    def get_db(cls):
        """Get database instance"""
        if cls._db is None:
            raise RuntimeError("MongoDB not initialized. Call initialize() first")
        
        # Add logging to help debug the type of object
        logger.info(f"MongoManager.get_db returning object of type: {type(cls._db).__name__}")
        
        # Check if _db has the expected methods of a RinDB instance
        if not hasattr(cls._db, 'store_tool_item_content'):
            logger.error("WARNING: _db instance doesn't have expected RinDB methods!")
            # Try to re-import RinDB and create a new instance
            try:
                from src.db.db_schema import RinDB
                if isinstance(cls._instance, AsyncIOMotorClient):
                    logger.info("Re-creating RinDB instance")
                    cls._db = RinDB(cls._instance)
                    logger.info(f"New _db type: {type(cls._db).__name__}")
            except Exception as e:
                logger.error(f"Failed to recreate RinDB instance: {e}")
                
        return cls._db

    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None
            cls._db = None
            logger.info("MongoDB connection closed")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if MongoDB is initialized"""
        return cls._instance is not None and cls._db is not None 