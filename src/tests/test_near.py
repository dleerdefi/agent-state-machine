import os
from dotenv import load_dotenv
import near_api
from near_api.account import Account
from near_api.signer import KeyPair, Signer
from near_api.providers import JsonProvider
import asyncio

async def test_near_connection():
    load_dotenv()
    account_id = os.getenv('NEAR_ACCOUNT_ID')
    private_key = os.getenv('NEAR_PRIVATE_KEY')
    rpc_url = os.getenv('NEAR_RPC_URL', 'https://rpc.mainnet.near.org')
    
    if not account_id or not private_key:
        print("Error: NEAR credentials not found in environment variables")
        return False
    
    try:
        # Initialize NEAR connection components
        provider = JsonProvider(rpc_url)
        key_pair = KeyPair(private_key)
        signer = Signer(account_id, key_pair)
        account = Account(provider, signer, account_id)
        
        # Test by checking NEAR balance
        balance = provider.get_account(account_id)
        print(f"✅ NEAR connection successful!")
        print(f"Account ID: {account_id}")
        print(f"Account Balance: {balance}")
        
        # Test a view function call to verify full functionality
        try:
            storage_balance = account.view_function(
                'wrap.near',  # well-known contract
                'storage_balance_of',
                {'account_id': account_id}
            )
            print(f"Storage Balance Check: {storage_balance}")
        except Exception as e:
            print(f"Note: Storage balance check failed (this is ok): {str(e)}")
            
        return True
    except Exception as e:
        print(f"❌ NEAR connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_near_connection()) 