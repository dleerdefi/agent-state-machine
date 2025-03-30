# Agent State Machine

A streamlined version of the RinAI project focused on the agent state machine, tools, and a simple chat interface.

## Features

- **Advanced State Machine Architecture**: Handle complex multi-turn interactions
- **Tool Integration**: Supports various tools including:
  - Twitter posting and scheduling
  - Cryptocurrency price data and limit orders
  - Weather information
  - Time and date conversion
  - Web search and summarization
  - NEAR Protocol intents and transactions
- **GraphRAG Memory**: Graph-based retrieval augmented generation for contextual responses
- **Simple Web Interface**: Chat with the agent through a browser-based interface

## Requirements

- Python 3.10+
- MongoDB
- Neo4j (optional, for GraphRAG)

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   MONGODB_URI=mongodb://localhost:27017/rinai
   NEAR_ACCOUNT_ID=your-near-account.testnet
   NEAR_PRIVATE_KEY=your-private-key
   NEAR_NETWORK=testnet
   COINGECKO_API_KEY=your-coingecko-api-key
   ```

3. Run the agent:
   ```
   python src/scripts/run_agent.py
   ```

4. Open the web interface:
   ```
   http://localhost:8766
   ```

## Configuration

You can configure the agent by creating a `config/config.json` file with the following structure:

```json
{
  "mongodb": {
    "uri": "mongodb://localhost:27017/rinai"
  },
  "near": {
    "account_id": "your-account.testnet",
    "private_key": "your-private-key",
    "network": "testnet"
  },
  "keys": [
    {
      "COINGECKO_API_KEY": "your-api-key"
    }
  ]
}
```

## License

MIT License - See LICENSE file for details 