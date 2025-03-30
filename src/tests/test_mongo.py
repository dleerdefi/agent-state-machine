import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio

async def test_mongo_connection():
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI')
    
    if not mongo_uri:
        print("Error: MONGO_URI not found in environment variables")
        return False
    
    try:
        client = AsyncIOMotorClient(mongo_uri)
        # Ping the database to test connection
        await client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_mongo_connection()) 