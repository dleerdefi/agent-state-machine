import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai
import asyncio

async def test_ai_connections():
    load_dotenv()
    results = {}
    
    # Test OpenAI
    try:
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10
        )
        results['OpenAI'] = "✅ Connected"
    except Exception as e:
        results['OpenAI'] = f"❌ Failed: {str(e)}"
    
    # Test Anthropic - Fix the async call
    try:
        client = anthropic.Anthropic()
        # Note: Anthropic's client is not async, so we don't use await
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello!"}]
        )
        results['Anthropic'] = "✅ Connected"
    except Exception as e:
        results['Anthropic'] = f"❌ Failed: {str(e)}"
    
    # Test Google AI - Fix the model name and method
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-1.0-pro')  # Updated model name
        response = model.generate_content(
            "Hello!",
            generation_config={"max_output_tokens": 10}
        )
        results['Google AI'] = "✅ Connected"
    except Exception as e:
        results['Google AI'] = f"❌ Failed: {str(e)}"
    
    # Print results
    print("\nAI Services Connection Test Results:")
    for service, status in results.items():
        print(f"{service}: {status}")

if __name__ == "__main__":
    asyncio.run(test_ai_connections()) 