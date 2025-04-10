import re
from typing import Dict, List, Optional
from src.db.enums import ToolType  # Add this import

class TriggerDetector:
    def __init__(self):
        # Initialize the tool_triggers dictionary first
        self.tool_triggers = {}
        
        # Define Twitter patterns once
        self.twitter_patterns = {
            'general': {
                'keywords': ['tweet', 'twitter', '@', 'post'],
                'phrases': [
                    'post on twitter', 'post on x', 'send a tweet', 'create a tweet', 'make a tweet',
                    'write a tweet', 'compose a tweet', 'publish a tweet', 'write a thread'
                ]
            },

            'schedule': {
                'keywords': [
                    'schedule', 'plan', 'series', 'multiple', 'timed', 'batch',
                    'queue', 'later', 'upcoming', 'future', 'tomorrow', 'next'
                ],
                'phrases': [
                    'schedule tweets', 'schedule a tweet', 'schedule two tweets', 'schedule three tweets', 'schedule four tweets', 'schedule five tweets', 'schedule six tweets', 'schedule seven tweets', 'schedule eight tweets', 'schedule nine tweets', 'schedule ten tweets', 'schedule 1 tweet', 'schedule 2 tweets', 'schedule 3 tweets', 'schedule 4 tweets', 'schedule 5 tweets', 'schedule 6 tweets', 'schedule 7 tweets', 'schedule 8 tweets', 'schedule 9 tweets', 'schedule 10 tweets', 'plan tweets', 'tweet series', 'queue tweets',
                    'queue up', 'line up', 'prepare tweets', 'post later', 'schedule for later', 'set up tweets',
                    'automate tweets', 'batch tweets'
                ]
            },
            'immediate': {
                'keywords': ['tweet now', 'post now', 'send tweet', 'create tweet', 'publish now', 'thread'],
                'phrases': [
                    'tweet this', 'post this', 'send this tweet', 'create a tweet', 'make a tweet',
                    'write a tweet thread', 'create a thread', 'start a thread'
                ]
            },
            'reply': {
                'keywords': ['reply', 'respond', 'comment'],
                'phrases': [
                    'reply to this', 'respond to this tweet', 'add a comment', 'reply to that'
                ]
            },
            'retweet': {
                'keywords': ['retweet', 'rt', 'share'],
                'phrases': [
                    'retweet this', 'quote tweet this', 'share this tweet'
                ]
            },
            'like': {
                'keywords': ['like', 'favorite'],
                'phrases': [
                    'like this tweet', 'favorite this', 'add to liked tweets'
                ]
            },
            'hashtag': {
                'keywords': ['#', 'hashtag'],
                'phrases': [
                    'add hashtag', 'include hashtag', 'use the hashtag'
                ]
            },
            'engagement': {
                'keywords': ['analytics', 'metrics', 'stats', 'engagement', 'reach'],
                'phrases': [
                    'how many likes', 'tweet stats', 'twitter engagement', 'show tweet performance'
                ]
            }
        }
        
        # Use ToolType enum values as keys for consistency
        self.tool_triggers[ToolType.CRYPTO_DATA.value] = {
            'keywords': [
                'bitcoin', 'btc', 'eth', 'solana', 'near', 'dogecoin', 'fartcoin', 'sui', 'ethereum', 'price', 'market', 'crypto', '$', 'coin',
                'token', 'altcoin', 'nft', 'blockchain', 'gas fee', 'wallet', 'swap', 'defi'
            ],
            'phrases': [
                'how much is', "what's the price", 'show me the market', 'crypto trends',
            ]
        }
        
        self.tool_triggers[ToolType.SEARCH.value] = {
            'keywords': [
                'news', 'latest', 'current', 'today', 'happened', 'recent', 'headline',
                'stocks', 'finance', 'updates', 'economy', 'elections', 'company', 'ceo',
                'IPO', 'merger', 'lawsuit', 'AI developments', 'search for', 'find info on', 'what is', 'tell me about'
            ],
            'phrases': [
                'what is happening', 'tell me about', 'what happened', 'search for',
                'explain this news', 'show me headlines', 'recent updates', 'trending stories'
            ]
        }

        # GraphRAG Memory triggers
        self.memory_triggers = {
            'keywords': ['remember', 'you said', 'earlier', 'before', 'last time', 'previously', 'we talked about', 'remind me'],
            'phrases': [
                'do you recall', 'as we discussed', 'like you mentioned', 'remind me what we said about',
                'bring up my last request', 'what did I say before', 'you told me earlier'
            ]
        }

        # Use ToolType enum values as keys for consistency
        self.tool_triggers[ToolType.TIME.value] = {
            'keywords': [
                'time', 'clock', 'timezone', 'tz', 'hour', 'date', 'schedule',
                'convert time', 'what time', 'current time'
            ],
            'phrases': [
                'what time is it', 'show me the time', 'current time in',
                'convert time from', 'time difference between'
            ]
        }

        self.tool_triggers[ToolType.WEATHER.value] = {
            'keywords': [
                'weather', 'temperature', 'forecast', 'rain', 'snow', 'humidity',
                'wind', 'precipitation', 'sunny', 'cloudy'
            ],
            'phrases': [
                'what\'s the weather', 'how\'s the weather', 'weather forecast',
                'is it going to rain', 'temperature in'
            ]
        }

        self.tool_triggers[ToolType.CALENDAR.value] = {
            'keywords': [
                'calendar', 'schedule', 'event', 'appointment', 'meeting',
                'agenda', 'upcoming', 'planned', 'booked', 'reminder'
            ],
            'phrases': [
                'what\'s on my calendar', 'show my schedule', 'upcoming events',
                'what do i have planned', 'check my calendar', 'what meetings do i have',
                'show my appointments', 'what\'s next on my schedule',
                'am i free', 'do i have any meetings'
            ]
        }

        self.tool_triggers[ToolType.INTENTS.value] = {
            'keywords': [
                'limit order', 'buy when', 'sell when', 'price reaches',
                'target price', 'execute when', 'trigger price', 'when price hits',
                'automated trade', 'conditional order', 'swap at'
            ],
            'phrases': [
                'create limit order', 'set up limit order', 'buy when price reaches',
                'sell when price reaches', 'execute trade at price', 'when price hits',
                'trade automatically at', 'set target price', 'create order to buy',
                'create order to sell', 'swap when price', 'exchange at price',
                'sell tokens when', 'buy tokens when', 'trade NEAR when', 'trade ETH when',
                'trade BTC when', 'trade USDC when', 'trade SOL when', 'trade USDT when'
            ]
        }

    def should_use_tools(self, message: str) -> bool:
        """Check if message should trigger tool usage"""
        message = message.lower()
        
        for tool_key in self.tool_triggers:
            tool = self.tool_triggers[tool_key]
            # Check keywords
            if any(keyword.lower() in message for keyword in tool['keywords']):
                return True
            # Check phrases
            if any(phrase.lower() in message for phrase in tool['phrases']):
                return True
                
        return False
        
    def should_use_memory(self, message: str) -> bool:
        """Check if message should trigger memory lookup"""
        message = message.lower()
        
        # Check memory keywords
        if any(keyword.lower() in message for keyword in self.memory_triggers['keywords']):
            return True
        # Check memory phrases
        if any(phrase.lower() in message for phrase in self.memory_triggers['phrases']):
            return True
            
        return False

    def should_use_twitter(self, message: str) -> bool:
        """Check if message is Twitter-related"""
        message = message.lower()
        
        # Check all Twitter pattern categories
        for category in self.twitter_patterns.values():
            if any(keyword.lower() in message for keyword in category['keywords']) or \
               any(phrase.lower() in message for phrase in category['phrases']):
                return True
                
        return False

    def get_tool_operation_type(self, message: str) -> Optional[str]:
        """Determine specific Twitter operation type from message"""
        message = message.lower()
        
        if not self.should_use_twitter(message):
            return None
            
        # Check for scheduling patterns first (most specific)
        if any(keyword in message for keyword in self.twitter_patterns['schedule']['keywords']) or \
           any(phrase in message for phrase in self.twitter_patterns['schedule']['phrases']):
            return "schedule_tweets"
            
        # Check for reply patterns
        if any(keyword in message for keyword in self.twitter_patterns['reply']['keywords']) or \
           any(phrase in message for phrase in self.twitter_patterns['reply']['phrases']):
            return "reply_tweet"
            
        # Check for retweet patterns
        if any(keyword in message for keyword in self.twitter_patterns['retweet']['keywords']) or \
           any(phrase in message for phrase in self.twitter_patterns['retweet']['phrases']):
            return "retweet"
            
        # Check for like patterns
        if any(keyword in message for keyword in self.twitter_patterns['like']['keywords']) or \
           any(phrase in message for phrase in self.twitter_patterns['like']['phrases']):
            return "like_tweet"
            
        # Check for immediate tweet patterns
        if any(keyword in message for keyword in self.twitter_patterns['immediate']['keywords']) or \
           any(phrase in message for phrase in self.twitter_patterns['immediate']['phrases']):
            return "send_tweet"
            
        # Check for engagement/analytics patterns
        if any(keyword in message for keyword in self.twitter_patterns['engagement']['keywords']) or \
           any(phrase in message for phrase in self.twitter_patterns['engagement']['phrases']):
            return "show_analytics"
            
        # Default to send_tweet for general Twitter actions
        return "send_tweet"

    def get_specific_tool_type(self, message: str) -> Optional[str]:
        """Determine specific tool type needed, prioritizing more specific triggers."""
        message = message.lower()
        
        # Prioritized Checks (Most specific first)
        # 1. Intents (Limit Orders)
        intents_key = ToolType.INTENTS.value
        if intents_key in self.tool_triggers and (any(keyword.lower() in message for keyword in self.tool_triggers[intents_key]['keywords']) or \
           any(phrase.lower() in message for phrase in self.tool_triggers[intents_key]['phrases'])):
            return intents_key
            
        # 2. Twitter (Complex internal logic)
        if self.should_use_twitter(message):
            return ToolType.TWITTER.value
            
        # 3. Weather
        weather_key = ToolType.WEATHER.value
        if weather_key in self.tool_triggers and (any(keyword.lower() in message for keyword in self.tool_triggers[weather_key]['keywords']) or \
           any(phrase.lower() in message for phrase in self.tool_triggers[weather_key]['phrases'])):
            return weather_key
            
        # 4. Time
        time_key = ToolType.TIME.value
        if time_key in self.tool_triggers and (any(keyword.lower() in message for keyword in self.tool_triggers[time_key]['keywords']) or \
           any(phrase.lower() in message for phrase in self.tool_triggers[time_key]['phrases'])):
            return time_key
            
        # 5. Crypto Data
        crypto_key = ToolType.CRYPTO_DATA.value
        if crypto_key in self.tool_triggers and (any(keyword.lower() in message for keyword in self.tool_triggers[crypto_key]['keywords']) or \
           any(phrase.lower() in message for phrase in self.tool_triggers[crypto_key]['phrases'])):
            return crypto_key
            
        # 6. Calendar
        calendar_key = ToolType.CALENDAR.value
        if calendar_key in self.tool_triggers and (any(keyword.lower() in message for keyword in self.tool_triggers[calendar_key]['keywords']) or \
           any(phrase.lower() in message for phrase in self.tool_triggers[calendar_key]['phrases'])):
            return calendar_key

        # General Checks (Less specific, checked last)
        # 7. Search (General web search)
        search_key = ToolType.SEARCH.value
        if search_key in self.tool_triggers and (any(keyword.lower() in message for keyword in self.tool_triggers[search_key]['keywords']) or \
           any(phrase.lower() in message for phrase in self.tool_triggers[search_key]['phrases'])):
            return search_key
            
        # Add checks for any other tools here if necessary
            
        return None # No specific tool detected 