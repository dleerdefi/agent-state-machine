#!/usr/bin/env python3
import os
import sys
import asyncio

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Monkey patch CoinGeckoClient before importing any modules that use it
import importlib.util
spec = importlib.util.spec_from_file_location(
    "coingecko_client", 
    os.path.join(current_dir, "src/clients/coingecko_client.py")
)
coingecko_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coingecko_module)

# Store the original class
original_client = coingecko_module.CoinGeckoClient

# Create a new class that accepts api_key=None
class PatchedCoinGeckoClient(original_client):
    def __init__(self, api_key=None):
        api_key = api_key or os.environ.get('COINGECKO_API_KEY')
        super().__init__(api_key=api_key)

# Replace the original class
coingecko_module.CoinGeckoClient = PatchedCoinGeckoClient
sys.modules['src.clients.coingecko_client'] = coingecko_module

# Now we can import the main function
from src.scripts.run_agent import main

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 