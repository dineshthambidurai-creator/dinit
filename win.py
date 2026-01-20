"""
Professional Trading System - Complete Version with Enhanced Fibonacci Analysis
Focus: Option Chain Analysis, Technical Indicators, Market Analysis, and Fibonacci Trend Detection
Fixed: Proper positions handling for 5Paisa API direct list response
Added: Comprehensive Fibonacci uptrend/downtrend analysis
Enhanced: High Accuracy Support/Resistance combining OI + Historical validation
"""

import json
import os
import sys
import time
import warnings
import sqlite3
import threading
import math
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field


import numpy as np
import pandas as pd
import pyotp
import ta
from py5paisa import FivePaisaClient

warnings.filterwarnings('ignore')

# ===============================
# UTILITY CLASSES
# ===============================

class SuppressPrints:
    """Context manager to suppress print statements."""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class Logger:
    """Simple logging utility - Fixed encoding issues."""
    @staticmethod
    def _log(level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp} | {level} | {message}", flush=True)
        
    @staticmethod
    def info(message: str) -> None:
        Logger._log("INFO", message)
    
    @staticmethod
    def success(message: str) -> None:
        Logger._log("SUCCESS", message)
    
    @staticmethod
    def error(message: str) -> None:
        Logger._log("ERROR", message)
    
    @staticmethod
    def warning(message: str) -> None:
        Logger._log("WARNING", message)

# ===============================
# CONFIGURATION
# ===============================

@dataclass
class TradingConfig:
    """Trading configuration constants."""
    SYMBOLS: List[str] = None
    STRIKE_STEPS: Dict[str, int] = None
    LOT_SIZES: Dict[str, int] = None 
    DATABASE_PATH: str = "trading_data.db"
    DATA_UPDATE_INTERVAL: int = 23  # seconds
    number_of_lots: int = 1
    oi_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)


    
    def __post_init__(self):
        
        if self.SYMBOLS is None:
            # self.SYMBOLS = ['NIFTY']
            # self.SYMBOLS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY',  'MIDCPNIFTY', 'SENSEX']
            self.SYMBOLS = ['NIFTY', 'BANKNIFTY']

        
        if self.STRIKE_STEPS is None:
            self.STRIKE_STEPS = {
                'NIFTY': 50,
                'BANKNIFTY': 100,
                'BANKEX': 100,
                'FINNIFTY': 50, 
                'SENSEX': 100,
                'MIDCPNIFTY': 50  
            }
            
        # ================= LOT SIZES (UPDATED – DEC 2025) =================
        if self.LOT_SIZES is None:
            self.LOT_SIZES = {
                'NIFTY': 65,
                'BANKNIFTY': 30,
                'FINNIFTY': 60,
                'MIDCPNIFTY': 120,
                'SENSEX': 20
            }

CONFIG = TradingConfig()

# ===============================
# DATABASE MANAGER
# ===============================


def new_trade_state():
    return {
        "active": False,
        "trade_id": None,
        "strike": None,
        "token": None,
        "entry_price": None,
        "qty": None,      
        "last_used_level": None,
        "entry_oi": None,
        "entry_delta": None
    }

class DatabaseManager:
    """Handles SQLite database operations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or CONFIG.DATABASE_PATH
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS option_chain_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        strike_price REAL NOT NULL,
                        option_type TEXT NOT NULL,
                        last_price REAL,
                        bid REAL,
                        ask REAL,
                        volume INTEGER,
                        open_interest INTEGER,
                        change_in_oi INTEGER,
                        implied_volatility REAL
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_option_chain_symbol_timestamp 
                    ON option_chain_data(symbol, timestamp, strike_price, option_type)
                """)
                
                # Market data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        open_price REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                    ON market_data(symbol, timestamp)
                """)
                # Add table for market analysis summary
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_analysis_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        current_price REAL,
                        market_open REAL,
                        market_high REAL,
                        market_low REAL,
                        market_volume INTEGER,
                        market_bias TEXT,
                        nearest_call_strike REAL,
                        nearest_call_last REAL,
                        nearest_put_strike REAL,
                        nearest_put_last REAL
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_analysis_summary_symbol_timestamp 
                    ON market_analysis_summary(symbol, timestamp)
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_option_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,

                        -- SENTIMENT TYPE
                        sentiment_type TEXT,      -- OI, OI_CHANGE, VOLUME

                        -- TRADE INFO
                        direction TEXT,           -- CE or PE
                        strike REAL,
                        token TEXT,
                        entry_price REAL,
                        exit_price REAL,

                        -- UNDERLYING PRICE
                        current_index_price REAL,

                        -- EXISTING MARKET DATA
                        market_open REAL,
                        market_high REAL,
                        market_low REAL,
                        market_volume REAL,

                        -- OPTION CHAIN RAW DATA
                        call_volume REAL,
                        put_volume REAL,
                        call_oi REAL,
                        put_oi REAL,
                        call_oi_change REAL,
                        put_oi_change REAL,

                        -- PCR DATA
                        pcr_oi REAL,
                        pcr_oi_change REAL,
                        pcr_volume REAL,
                        pcr_vol_price REAL,

                        -- SENTIMENT VALUES
                        oi_sentiment TEXT,
                        oi_change_sentiment TEXT,
                        volume_sentiment TEXT,
                        combined_sentiment TEXT
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_option_trades_symbol_timestamp 
                    ON sentiment_option_trades(symbol, timestamp)
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS delta_oi_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,

                    trade_side TEXT,              -- BUY_CE / BUY_PE
                    oi_sentiment TEXT,            -- BULLISH / BEARISH

                    entry_price REAL,
                    current_price REAL,
                    exit_price REAL,

                    pcr_oi REAL,
                    delta_bias_score REAL,
                    gamma_flip_level REAL,

                    status TEXT                   -- OPEN / CLOSED
                )
                """)

                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_delta_oi_trades_symbol_timestamp
                ON delta_oi_trades(symbol, timestamp)
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS option_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    symbol TEXT NOT NULL,
                    strategy TEXT,
                    option_type TEXT,

                    strike REAL,
                    token TEXT,
                    qty INTEGER,

                    entry_price REAL,
                    current_price REAL,
                    exit_price REAL,
                    pnl REAL,

                    status TEXT,

                    entry_oi TEXT,
                    entry_delta REAL,

                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    exit_time DATETIME
                )
                """)
                
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_option_trades_symbol_status
                ON option_trades(symbol, status)
                """)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_option_trades_strategy
                ON option_trades(strategy)
                """)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_option_trades_entry_time
                ON option_trades(entry_time)
                """)

                conn.commit()
                
        except Exception as e:
            Logger.error(f"Failed to initialize database: {e}")
            raise

    def store_market_analysis_summary(self, symbol: str, current_price: float, current_data: Dict, market_bias: str, nearest_call: Dict, nearest_put: Dict) -> None:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    timestamp = datetime.now()

                    cursor.execute("""
                        INSERT INTO market_analysis_summary (
                            timestamp, symbol, current_price, market_open, market_high, market_low,
                            market_volume, market_bias, nearest_call_strike, nearest_call_last,
                            nearest_put_strike, nearest_put_last
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, symbol, current_price,
                        current_data.get('open', 0), current_data.get('high', 0), current_data.get('low', 0),
                        current_data.get('volume', 0), market_bias,
                        nearest_call.get('strike_price', 0) if nearest_call else None,
                        nearest_call.get('last_price', 0) if nearest_call else None,
                        nearest_put.get('strike_price', 0) if nearest_put else None,
                        nearest_put.get('last_price', 0) if nearest_put else None
                    ))

                    conn.commit()
            except Exception as e:
                Logger.error(f"Failed to store market analysis summary: {e}")

    def store_sentiment_trade(self,symbol: str,sentiment_type: str,direction: str,strike: float,token: str,entry_price: float,exit_price: float,current_index_price: float,current_data: Dict,option_analysis: Dict) -> None:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        timestamp = datetime.now()

                        cursor.execute("""
                        INSERT INTO sentiment_option_trades (
                            timestamp, symbol, sentiment_type, direction, strike, token,
                            entry_price, exit_price, current_index_price,

                            market_open, market_high, market_low, market_volume,

                            call_volume, put_volume, call_oi, put_oi,
                            call_oi_change, put_oi_change,

                            pcr_oi, pcr_oi_change, pcr_volume, pcr_vol_price,

                            oi_sentiment, oi_change_sentiment, volume_sentiment, combined_sentiment,

                            avg_iv, iv_rank, iv_percentile,
                            gamma_flip_level, delta_bias_score,
                            sl_points, target_points, trade_signal
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp, symbol, sentiment_type, direction, strike, token,
                            entry_price, exit_price, current_index_price,

                            current_data.get('open', 0),
                            current_data.get('high', 0),
                            current_data.get('low', 0),
                            current_data.get('volume', 0),

                            option_analysis.get('call_volume', 0),
                            option_analysis.get('put_volume', 0),
                            option_analysis.get('call_oi', 0),
                            option_analysis.get('put_oi', 0),
                            option_analysis.get('call_oi_change', 0),
                            option_analysis.get('put_oi_change', 0),

                            option_analysis.get('pcr_oi', 0),
                            option_analysis.get('pcr_oi_change', 0),
                            option_analysis.get('pcr_volume', 0),
                            option_analysis.get('pcr_vol_price', 0),

                            option_analysis.get('oi_sentiment', 'N/A'),
                            option_analysis.get('oi_change_sentiment', 'N/A'),
                            option_analysis.get('volume_sentiment', 'N/A'),
                            option_analysis.get('sentiment', 'N/A'),

                            option_analysis.get('avg_iv', 0),
                            option_analysis.get('iv_rank', 0),
                            option_analysis.get('iv_percentile', 0),

                            option_analysis.get('gamma_flip_level', 0),
                            option_analysis.get('delta_bias_score', 0),

                            option_analysis.get('sl_points', 0),
                            option_analysis.get('target_points', 0),
                            option_analysis.get('trade_signal', None)
                        ))


                        conn.commit()                    
            except Exception as e:
                Logger.error(f"Failed to store sentiment trade: {e}")
       
    def store_option_chain_data(self, symbol: str, option_chain_data: List[Dict]) -> None:
        """Store option chain data."""
        if not option_chain_data:
            return
            
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    timestamp = datetime.now()
                    
                    for option in option_chain_data:
                        cursor.execute("""
                            INSERT INTO option_chain_data (
                                timestamp, symbol, strike_price, option_type,
                                last_price, bid, ask, volume, open_interest, 
                                change_in_oi, implied_volatility
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp, symbol, option.get('strike_price', 0),
                            option.get('option_type', ''), option.get('last_price', 0),
                            option.get('bid', 0), option.get('ask', 0),
                            option.get('volume', 0), option.get('open_interest', 0),
                            option.get('change_in_oi', 0), option.get('implied_volatility', 0)
                        ))
                    
                    conn.commit()
                    # Logger.success(f"Stored {len(option_chain_data)} option chain records for {symbol}")
                    
            except Exception as e:
                Logger.error(f"Failed to store option chain data: {e}")
    
    def store_market_data(self, symbol: str, market_data: Dict) -> None:
        """Store market data."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    timestamp = datetime.now()
                    
                    cursor.execute("""
                        INSERT INTO market_data (
                            timestamp, symbol, open_price, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, symbol, market_data.get('open', 0),
                        market_data.get('high', 0), market_data.get('low', 0),
                        market_data.get('close', 0), market_data.get('volume', 0)
                    ))
                    
                    conn.commit()
                    
            except Exception as e:
                Logger.error(f"Failed to store market data: {e}")

    def store_delta_oi_trade(
        self,
        symbol: str,
        trade_side: str,
        oi_sentiment: str,
        entry_price: float,
        current_price: float,
        exit_price: float,
        pcr_oi: float,
        delta_bias_score: float,
        gamma_flip_level: float,
        status: str
    ):
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO delta_oi_trades (
                            symbol, trade_side, oi_sentiment,
                            entry_price, current_price, exit_price,
                            pcr_oi, delta_bias_score, gamma_flip_level,
                            status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        trade_side,
                        oi_sentiment,
                        entry_price,
                        current_price,
                        exit_price,
                        pcr_oi,
                        delta_bias_score,
                        gamma_flip_level,
                        status
                    ))
                    conn.commit()
            except Exception as e:
                Logger.error(f"Failed to store delta OI trade: {e}")


    def insert_trade(self, **data):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO option_trades (
                    symbol, strategy, option_type, strike, token,
                    qty, entry_price, status,
                    entry_oi, entry_delta, entry_time
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                data["symbol"], data["strategy"], data["option_type"],
                data["strike"], data["token"],
                data["qty"], data["entry_price"],
                "OPEN", data["entry_oi"], data["entry_delta"],
                datetime.now().isoformat()
            ))
            return cur.lastrowid


    def close_trade(self, trade_id, exit_price, pnl):
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                UPDATE option_trades SET
                    exit_price=?,
                    pnl=?,
                    status='CLOSED',
                    exit_time=?
                WHERE id=?
            """, (exit_price, pnl, datetime.now().isoformat(), trade_id))

# ===============================
# API CLIENT
# ===============================

class APIClient:
    """Handles API client initialization and data fetching."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.client: Optional[FivePaisaClient] = None
        self.credentials_path = credentials_path
        self._scrips_cache: Optional[pd.DataFrame] = None
        self._open_fix_state: Dict[str, bool] = {}
        self._daily_open_cache: Dict[str, bool] = {}

    def _load_credentials(self) -> dict:

        # ✅ LOCAL FILE MODE
        if self.credentials_path and os.path.exists(self.credentials_path):
            with open(self.credentials_path, "r") as f:
                cred = json.load(f)

        # ✅ ENV / GITHUB ACTIONS MODE
        else:
            cred = {
                "CLIENT_CODE": os.getenv("CLIENT_CODE"),
                "PIN": os.getenv("PIN"),
                "TOTP_SECRET": os.getenv("TOTP_SECRET"),
                "APP_NAME": os.getenv("APP_NAME"),
                "APP_SOURCE": os.getenv("APP_SOURCE"),
                "USER_ID": os.getenv("USER_ID"),
                "PASSWORD": os.getenv("PASSWORD"),
                "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY"),
                "USER_KEY": os.getenv("USER_KEY")
            }

        # ✅ VALIDATION (MANDATORY)
        missing = [k for k, v in cred.items() if not v]
        if missing:
            raise RuntimeError(f"Missing 5Paisa credentials: {missing}")

        return cred
    def initialize_client(self) -> bool:
        """Initialize 5Paisa client with TOTP authentication."""
        
        try:
            cred = self._load_credentials()
            
            client = FivePaisaClient(cred=cred)
            totp = pyotp.TOTP(cred["TOTP_SECRET"])
            
            # Ensure TOTP timing
            remaining = 30 - (int(time.time()) % 30)
            if remaining < 5:
                time.sleep(remaining + 1)
            
            current_totp = totp.now()
            
            with SuppressPrints():
                session = client.get_totp_session(
                    client_code=cred["CLIENT_CODE"],
                    totp=current_totp,
                    pin=cred["PIN"]
                )
            
            if session:
                with SuppressPrints():
                    self.client = client
                    # Logger.success("API client initialized successfully")
                    return True
            else:
                Logger.error("Failed to create session")
                return False
            
        except Exception as e:
            Logger.error(f"Failed to initialize API client: {e}")
            return False
    
    def load_scrips_data(self, file_path: str = "scrips_data.json") -> Optional[pd.DataFrame]:
        """Load scrips from cache or fetch from API."""
        if self._scrips_cache is not None:
            return self._scrips_cache
            
        try:
            # Try loading from cache first
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    scrips_data = json.load(f)
                self._scrips_cache = pd.DataFrame(scrips_data)
                # Logger.success(f"Loaded {len(self._scrips_cache)} scrips from cache")
                return self._scrips_cache
            
            # Fetch from API if no cache
            if not self.client:
                # Logger.error("API client not initialized")
                return None
                
            with SuppressPrints():
                scrips_live = self.client.get_scrips()
            
            if scrips_live is not None and not scrips_live.empty:
                scrips_live.to_json(file_path, orient='records', indent=4)
                self._scrips_cache = scrips_live
                # Logger.success(f"Fetched {len(scrips_live)} scrips from API")
                return scrips_live
                
        except Exception as e:
            Logger.error(f"Failed to load scrips data: {e}")
            
        return None
    
    def find_scrip_info(self, symbol: str) -> Optional[pd.Series]:
        """Find scrip information for a given symbol."""
        scrips_df = self.load_scrips_data()
        if scrips_df is None or scrips_df.empty:
            return None
        
        # Direct match strategy
        for exch in ['N', 'B']:  # NSE
            for exch_type in ['I', 'C', 'D']:  # Index, Cash, Derivatives
                direct_match = scrips_df[
                    (scrips_df['Name'].str.upper() == symbol.upper()) &
                    (scrips_df['Exch'] == exch) &
                    (scrips_df['ExchType'] == exch_type)
                ]
                
                if not direct_match.empty:
                    return direct_match.iloc[0]
        
        return None
    

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current market price and OHLC data for a symbol."""
        scrip_info = self.find_scrip_info(symbol)
        if scrip_info is None:
            Logger.error(f"Scrip info not found for {symbol}")
            return None

        try:
            req_list = [{
                "Exch": scrip_info['Exch'],
                "ExchType": scrip_info['ExchType'],
                "ScripCode": int(scrip_info['ScripCode'])
            }]

            with SuppressPrints():
                market_data = self.client.fetch_market_feed_scrip(req_list)

            if market_data and 'Data' in market_data and len(market_data['Data']) > 0:
                price_data = market_data['Data'][0]

                # -------------------------------
                # TIME CHECK (IST assumed system time)
                # -------------------------------
                now = datetime.now()
                after_914 = (now.hour > 9) or (now.hour == 9 and now.minute >= 14)
                today_key = f"{symbol}_{now.date()}"
                ltp = float(price_data.get('LastRate', 0))
                raw_open = float(price_data.get('LastRate', 0))

                # -------------------------------
                # FIX OPEN ONLY ONCE PER DAY
                # -------------------------------
                if (
                    after_914
                    and not self._open_fix_state.get(today_key)
                    and ltp > 0
                ):
                    open_price = ltp
                    self._open_fix_state[today_key] = True
                else:
                    open_price = float(raw_open or 0)
                


                return {
                    'open': open_price,
                    'high': float(price_data.get('High', 0)),
                    'low': float(price_data.get('Low', 0)),
                    'close': ltp,
                    'pclose': float(price_data.get('PClose', 0)),
                    'volume': int(price_data.get('Volume', 0)),
                    'ltp': ltp,
                    'Exch': price_data.get('Exch', 'N')
                }

            else:
                Logger.error(f"No market data received for {symbol}")

        except Exception as e:
            Logger.error(f"Failed to get price for {symbol}: {e}")

        return None

    from datetime import datetime

    # def get_current_price(self, symbol: str):
    #     """Get current market price and OHLC data for a symbol."""
    #     scrip_info = self.find_scrip_info(symbol)
    #     if scrip_info is None:
    #         Logger.error(f"Scrip info not found for {symbol}")
    #         return None

    #     try:
    #         req_list = [{
    #             "Exch": scrip_info['Exch'],
    #             "ExchType": scrip_info['ExchType'],
    #             "ScripCode": int(scrip_info['ScripCode'])
    #         }]

    #         with SuppressPrints():
    #             market_data = self.client.fetch_market_feed_scrip(req_list)

    #         if not market_data or 'Data' not in market_data or not market_data['Data']:
    #             Logger.error(f"No market data received for {symbol}")
    #             return None

    #         price_data = market_data['Data'][0]
    #         ltp = float(price_data.get('LastRate', 0))

    #         # --------------------------------
    #         # DAILY OPEN — FETCH ONLY ONCE
    #         # --------------------------------
    #         today = datetime.now().date()
    #         open_key = f"{symbol}_{today}"

    #         if open_key not in self._daily_open_cache:
    #             historical_1d = self.fetch_historical_data(
    #                 symbol,
    #                 mins="1d",
    #                 days_back=1
    #             )

    #             open_price = ltp

    #             # ✅ DATAFRAME SAFE CHECK
    #             if historical_1d is not None and not historical_1d.empty:
    #                 # Assumes columns: ['open','high','low','close','volume']
    #                 open_price = float(historical_1d.iloc[0]["Open"])

    #             self._daily_open_cache[open_key] = open_price

    #         else:
    #             open_price = self._daily_open_cache[open_key]

    #         return {
    #             "open": open_price,                     # ✅ cached daily open
    #             "high": float(price_data.get("High", 0)),
    #             "low": float(price_data.get("Low", 0)),
    #             "close": ltp,
    #             "pclose": float(price_data.get("PClose", 0)),
    #             "volume": int(price_data.get("Volume", 0)),
    #             "ltp": ltp,
    #             "Exch": price_data.get("Exch", "N"),
    #         }

    #     except Exception as e:
    #         Logger.error(f"Failed to get price for {symbol}: {e}")

    #     return None

    def is_trading_time(self) -> bool:
        """
        Returns True if current time is within NSE trading hours
        (Mon–Fri, 09:15–15:30)
        """
        now = datetime.now()

        # Weekend check
        # if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        #     return False
        market_open = dtime(9, 28)
        market_close = dtime(22, 30)
        return market_open <= now.time() <= market_close

    def is_live_entry_time(self):
        now = datetime.now().time()
        return dtime(9, 30) <= now <= dtime(15, 0)

    def is_force_exit_time(self):
        now = datetime.now().time()
        return dtime(15, 5) <= now <= dtime(15, 15)

    def get_market_status(self) -> dict:
        """
        Check market status ONLY for:
        Exch = N (NSE)
        ExchType = C (Cash)
        """

        market_status = {
            "is_open": False,
            "message": "Market is closed"
        }

        # Call 5paisa SDK
        market_status_api = self.client.get_market_status()

        # SDK returns a LIST of exchange statuses
        for exch in market_status_api:
            if exch.get("Exch") == "N" and exch.get("ExchType") == "C":
                if exch.get("MarketStatus") == "Open":
                    market_status["is_open"] = True
                    market_status["message"] = "NSE Cash market is OPEN"
                else:
                    market_status["message"] = "NSE Cash market is CLOSED"
                break

        return market_status


    def fetch_historical_data(self, symbol: str, mins: str, days_back: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol."""
        scrip_info = self.find_scrip_info(symbol)
        if scrip_info is None:
            Logger.error(f"Scrip info not found for {symbol}")
            return None
        
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            with SuppressPrints():
                hist_data = self.client.historical_data(
                    Exch=scrip_info['Exch'],
                    ExchangeSegment=scrip_info['ExchType'],
                    ScripCode=int(scrip_info['ScripCode']),
                    time=mins,  # 5-minute data for better analysis
                    From=start_date,
                    To=end_date
                )
            
            if hist_data is not None and not hist_data.empty:
                # Logger.success(f"Fetched {len(hist_data)} historical records for {symbol}")
                return hist_data
            else:
                Logger.error(f"No historical data received for {symbol}")
                return None
            
        except Exception as e:
            Logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None
    
    def get_option_chain_data(self, symbol: str) -> List[Dict]:
        """Get option chain data from API."""
        if not self.client:
            Logger.error("API client not initialized")
            return []
            
        try:
            # with SuppressPrints():
                # Get expiry dates
                scrip_info = self.find_scrip_info(symbol)
                expiry_data = self.client.get_expiry(scrip_info['Exch'] , symbol)
                
                if not expiry_data or 'Expiry' not in expiry_data or not expiry_data['Expiry']:
                    Logger.error(f"No expiry data for {symbol}")
                    return []
                
                # Get nearest expiry
                expiry_timestamp = None
                expiry_date_str = expiry_data['Expiry'][0]['ExpiryDate']
                
                import re
                match = re.search(r'/Date\((\d+)', expiry_date_str)
                if match:
                    expiry_timestamp = int(match.group(1))
                
                if not expiry_timestamp:
                    Logger.error(f"Could not parse expiry timestamp for {symbol}")
                    return []
                
                # Get option chain
                option_chain = self.client.get_option_chain(scrip_info['Exch'] , symbol, expiry_timestamp)
                if isinstance(option_chain, dict):
                    for key in ['Options', 'Data', 'OptionChain']:
                        if key in option_chain:
                            processed_data = self._process_option_chain_data(option_chain[key],expiry_timestamp=expiry_timestamp)
                            # Logger.success(f"Fetched option chain for {symbol}: {len(processed_data)} contracts")
                            return processed_data
                elif hasattr(option_chain, 'to_dict'):
                    processed_data = self._process_option_chain_data(option_chain.to_dict('records'),expiry_timestamp=expiry_timestamp)
                    # Logger.success(f"Fetched option chain for {symbol}: {len(processed_data)} contracts")
                    return processed_data
                    
        except Exception as e:
            Logger.error(f"Failed to get option chain for {symbol}: {e}")
            
        return []
    
    def _process_option_chain_data(self, raw_data: List[Dict],expiry_timestamp:datetime) -> List[Dict]:
        """Process raw option chain data into standardized format."""
        processed_data = []
        
        for option in raw_data:
            try:
                processed_data.append({                    
                    'name': option.get('Name', ''),
                    'ScripCode': int(option.get('ScripCode', 0)),
                    'strike_price': float(option.get('StrikeRate', 0)),
                    'option_type': option.get('CPType', ''),
                    'last_price': float(option.get('LastRate', 0)),
                    'bid': float(option.get('BidPrice', 0)),
                    'ask': float(option.get('AskPrice', 0)),
                    'volume': int(option.get('Volume', 0)),
                    'open_interest': int(option.get('OpenInterest', 0)),
                    'change_in_oi': int(option.get('ChangeInOI', 0)),
                    'implied_volatility': float(option.get('IV', 0)),
                    'expiry': expiry_timestamp
                })
            except (ValueError, TypeError) as e:
                Logger.error(f"Error processing option data: {e}")
                continue
                
        return processed_data
    
    def get_positions(self) -> List[Dict]:
        """Fetch current positions using 5Paisa API client - FIXED VERSION."""
        if not self.client:
            Logger.error("API client not initialized")
            return []
            
        try:
            with SuppressPrints():
                positions = self.client.positions()
                        
            if positions and isinstance(positions, list):
                return positions
            else:
                return []
                
        except Exception as e:
            Logger.error(f"Failed to fetch positions: {e}")
            return []

    def place_order_api(self, scripCode, direction, quantity, price, strike_price=None, option_type=None, exchange=None):
        """Enhanced order placement function for 5Paisa API."""
        try:
            order_type = 'B' if direction.upper() == 'BUY' else 'S'
            
            order_response = self.client.place_order(
                OrderType=order_type,
                Exchange=exchange,
                ExchangeType='D',  # Derivatives
                ScripCode=int(scripCode),
                Qty=quantity,
                Price=price,
                IsIntraday=False,  # Set to True for intraday
                DisQty=0,
                StopLossPrice=0,
                IsVTD=False,
                IOCOrder=False,
                IsAHPlaced=exchange
            )
            
            order_details = f"{direction} {quantity} {scripCode}"
            if strike_price and option_type:
                order_details += f" {strike_price} {option_type}"
            order_details += f" @ Rs.{price:.2f}"
            
            print(f"ORDER PLACED: {order_details}")
            return True
            
        except Exception as e:
            print(f"Order placement failed: {e}")
            return False

# ===============================
# TECHNICAL INDICATORS
# ===============================

class TechnicalIndicators:
    """Technical indicators calculation."""
    
    @staticmethod
    def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        try:
            # Moving averages
            df['SMA_5'] = ta.trend.sma_indicator(df["Close"], window=5)
            df['SMA_10'] = ta.trend.sma_indicator(df["Close"], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df["Close"], window=20)
            df['EMA_9'] = ta.trend.ema_indicator(df["Close"], window=9)
            df['EMA_21'] = ta.trend.ema_indicator(df["Close"], window=21)
            df['EMA_50'] = ta.trend.ema_indicator(df["Close"], window=50)
            
            # VWAP
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Momentum indicators
            df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
            df["MACD"] = ta.trend.macd_diff(df["Close"])
            df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
            
            # Volatility indicators
            df["BB_High"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
            df["BB_Low"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
            
            # Stochastic
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
            
            # Williams %R
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
            
        except Exception as e:
            Logger.error(f"Error computing technical indicators: {e}")
            
        return df

# ===============================
# MARKET ANALYSIS
# ===============================
from collections import deque
from statistics import mean
from typing import List, Dict, Any
import math
from datetime import datetime
from scipy.stats import norm
class MarketAnalyzer:
    """Market analysis functions."""
    
    pcr_history: Dict[str,  Dict[str, Any]] = {}

    @staticmethod
    def _bs_price(S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == "CE":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


    @staticmethod
    def calculate_implied_volatility(price, S, K, T, r, option_type):
        if price <= 0 or T <= 0:
            return 0.0

        sigma = 0.30
        for _ in range(50):
            bs_price = MarketAnalyzer._bs_price(S, K, T, r, sigma, option_type)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T)

            if vega == 0:
                break

            diff = bs_price - price
            if abs(diff) < 1e-4:
                break

            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5))

        return float(round(sigma * 100, 2))
    
    @staticmethod
    def calculate_greeks(S, K, T, r, iv, option_type):
        if T <= 0 or iv <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

        sigma = iv / 100
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        delta = norm.cdf(d1) if option_type == "CE" else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100

        if option_type == "CE":
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

        return {
            "delta": float(round(delta, 4)),
            "gamma": float(round(gamma, 6)),
            "theta": float(round(theta, 2)),
            "vega": float(round(vega, 4))
        }


    @staticmethod
    def find_support_resistance_option_chain(
                option_chain_data: List[Dict],
                current_price: float,
                pclose_price: float,
                window: float = 800.0
            ) -> Tuple[List[float], List[float], str]:
                """
                Find dynamic support and resistance levels based on
                Option Chain Open Interest, Change in OI, and Volume.

                Returns:
                    supports:    Top 3 strong PUT strikes (support)
                    resistances: Top 3 strong CALL strikes (resistance)
                    market_bias: Bullish / Bearish text label
                """
                if not option_chain_data:
                    return [], [], "No data"

                # Separate Calls and Puts with valid OI
                calls = [opt for opt in option_chain_data
                        if opt.get("option_type") == "CE" and opt.get("open_interest", 0) > 0]
                puts = [opt for opt in option_chain_data
                        if opt.get("option_type") == "PE" and opt.get("open_interest", 0) > 0]

                if not calls and not puts:
                    return [], [], "No valid OI"

                # Filter by window around previous close
                put_candidates = [
                    opt for opt in puts
                    if pclose_price - window <= opt.get("strike_price", 0) <= pclose_price
                ]
                call_candidates = [
                    opt for opt in calls
                    if pclose_price <= opt.get("strike_price", 0) <= pclose_price + window
                ]

                # Handle edge cases
                if not put_candidates and not call_candidates:
                    return [], [], "No strikes in window"

                max_put_oi = max((opt.get("open_interest", 0) for opt in put_candidates), default=0)
                max_call_oi = max((opt.get("open_interest", 0) for opt in call_candidates), default=0)

                def calculate_strength(opt: Dict, atm_price: float, side_max_oi: float) -> float:
                    """
                    High-win-rate strength formula.
                    Filters out weak OI, caps extreme volume, gives heavy ATM weight.
                    """
                    oi = float(opt.get("open_interest", 0.0))
                    if side_max_oi > 0 and oi < side_max_oi * 0.25:
                        return 0.0

                    change_oi = float(opt.get("change_in_oi", 0.0))
                    # Ignore unwinding
                    if change_oi < 0:
                        change_oi = 0.0

                    volume = float(opt.get("volume", 0.0))
                    # Cap fake/liquidity spikes
                    if oi > 0:
                        volume = min(volume, oi * 2.0)

                    strike = float(opt.get("strike_price", 0.0))
                    distance_from_atm = abs(strike - atm_price)

                    # Dynamic ATM dominance: closer levels get big advantage
                    atm_weight = max(1.0, 200.0 / (1.0 + distance_from_atm))

                    # 🔥 PRO High Accuracy Strength Formula
                    strength = (
                        (oi * 0.50) +          # Writer commitment
                        (change_oi * 0.35) +   # Fresh positioning / momentum
                        (volume * 0.10) +      # Participation filter
                        (atm_weight * 0.55)    # Zone dominance (proximity to ATM)
                    )

                    return strength

                # Compute PUT side strengths (Support)
                put_levels = []
                for opt in put_candidates:
                    strength = calculate_strength(opt, current_price, max_put_oi)
                    if strength <= 0:
                        continue
                    put_levels.append((opt.get("strike_price"), strength))

                put_levels.sort(key=lambda x: x[1], reverse=True)
                supports = [level[0] for level in put_levels[:3]]

                # Compute CALL side strengths (Resistance)
                call_levels = []
                for opt in call_candidates:
                    strength = calculate_strength(opt, current_price, max_call_oi)
                    if strength <= 0:
                        continue
                    call_levels.append((opt.get("strike_price"), strength))

                call_levels.sort(key=lambda x: x[1], reverse=True)
                resistances = [level[0] for level in call_levels[:3]]

                # Bias detection
                total_call_strength = sum(level[1] for level in call_levels)
                total_put_strength = sum(level[1] for level in put_levels)

                if total_put_strength == 0 and total_call_strength == 0:
                    market_bias = "Neutral / No clear writers"
                elif total_put_strength >= total_call_strength:
                    market_bias = "🟢 Bullish Bias (Put writers dominant)"
                else:
                    market_bias = "🔴 Bearish Bias (Call writers dominant)"

                return supports, resistances, market_bias
    
    
    @staticmethod
    def analyze_option_chain(option_chain_data: List[Dict], current_price: float,
                             pclose_price: float, symbol: str, window: float = 800) -> Dict[str, Any]:
        
        # ======================================================
        # 🔐 SYMBOL-SCOPED STATE (FULL INITIALIZATION)
        # ======================================================
        hist = MarketAnalyzer.pcr_history.setdefault(symbol, {
            "oi": [],
            "oi_change": [],
            "volume": [],
            "iv_history": [],

            "oi_sentiment": "NEUTRAL",
            "oi_change_sentiment": "NEUTRAL",
            "volume_sentiment": "NEUTRAL",
            "combined_sentiment": "NEUTRAL",

            "oi_trade": None,
            "oi_strike": None,
            "oi_token": None,
            "oi_entry_price": None,

            "oi_chg_trade": None,
            "oi_chg_strike": None,
            "oi_chg_token": None,
            "oi_chg_entry_price": None,

            "vol_trade": None,
            "vol_strike": None,
            "vol_token": None,
            "vol_entry_price": None
        })

        if not option_chain_data:
            return {}
        # ----------------------------------------------------
        # A) DYNAMIC EXPIRY DETECTION (SAFE FOR STR / INT / DATETIME)
        # ----------------------------------------------------
        from datetime import datetime

        def _parse_expiry(val):
            if isinstance(val, datetime):
                return val

            if isinstance(val, str):
                # expected format: YYYY-MM-DD
                return datetime.strptime(val, "%Y-%m-%d")

            if isinstance(val, (int, float)):
                # detect milliseconds vs seconds
                if val > 1e12:      # milliseconds
                    return datetime.fromtimestamp(val / 1000)
                else:               # seconds
                    return datetime.fromtimestamp(val)

            raise ValueError(f"Unsupported expiry format: {val}")

        expiry_dates = [opt.get("expiry") for opt in option_chain_data if opt.get("expiry")]

        if expiry_dates:
            expiry_dt = min(_parse_expiry(e) for e in expiry_dates)

            # Time to expiry in YEARS (use seconds, not days)
            T = max(
                (expiry_dt - datetime.now()).total_seconds(),
                60  # minimum 1 minute to avoid zero
            ) / (365 * 24 * 60 * 60)
        else:
            # fallback: assume weekly expiry
            T = 5 / 365

        r = 0.06  # risk-free rate (India)


        # ----------------------------------------------------
        # B) CALCULATE IV + GREEKS FOR EACH OPTION
        # ----------------------------------------------------
        for opt in option_chain_data:
            iv = MarketAnalyzer.calculate_implied_volatility(
                opt.get("last_price", 0),
                current_price,
                opt.get("strike_price", 0),
                T, r,
                opt.get("option_type")
            )

            greeks = MarketAnalyzer.calculate_greeks(
                current_price,
                opt.get("strike_price", 0),
                T, r, iv,
                opt.get("option_type")
            )

            opt["implied_volatility"] = iv
            opt.update(greeks)

        # ----------------------------------------------------
        # C) IV RANK / PERCENTILE
        # ----------------------------------------------------
        iv_vals = [o["implied_volatility"] for o in option_chain_data if o["implied_volatility"] > 0]
        avg_iv = round(sum(iv_vals) / len(iv_vals), 2) if iv_vals else 0

        hist["iv_history"].append(avg_iv)

        iv_min = min(hist["iv_history"])
        iv_max = max(hist["iv_history"])

        iv_rank = round(((avg_iv - iv_min) / (iv_max - iv_min)) * 100, 2) if iv_max > iv_min else 0
        iv_percentile = round(
            sum(iv <= avg_iv for iv in hist["iv_history"]) / len(hist["iv_history"]) * 100, 2
        )

        # ----------------------------------------------------
        # D) GAMMA FLIP LEVEL
        # ----------------------------------------------------
        gamma_by_strike = {}
        for opt in option_chain_data:
            strike = opt.get("strike_price")
            gamma_by_strike[strike] = gamma_by_strike.get(strike, 0) + opt.get("gamma", 0)

        gamma_flip_level = min(
            gamma_by_strike,
            key=lambda k: abs(k - current_price)
        ) if gamma_by_strike else None

        # ----------------------------------------------------
        # E) DELTA-NEUTRAL BIAS SCORE (0–100)
        # ----------------------------------------------------
        net_delta = sum(
            opt.get("delta", 0) * opt.get("open_interest", 0)
            for opt in option_chain_data
        )

        delta_bias_score = round(50 + max(min(net_delta / 1e6, 50), -50), 2)


        # ----------------------------------------------------
        # F) GREEKS-BASED SL / TARGET
        # ----------------------------------------------------
        avg_gamma = sum(o.get("gamma", 0) for o in option_chain_data) / max(len(option_chain_data), 1)

        sl_points = round(max(20, current_price * avg_gamma * 0.8), 2)
        target_points = round(sl_points * 1.8, 2)

        # ----------------------------------------------------
        # 1) Separate CE / PE
        # ----------------------------------------------------
        calls = [opt for opt in option_chain_data if opt.get("option_type") == "CE"]
        puts  = [opt for opt in option_chain_data if opt.get("option_type") == "PE"]

        call_volume    = sum(opt.get("volume", 0) for opt in calls)
        put_volume     = sum(opt.get("volume", 0) for opt in puts)

        call_oi        = sum(opt.get("open_interest", 0) for opt in calls)
        put_oi         = sum(opt.get("open_interest", 0) for opt in puts)

        call_oi_change = sum(opt.get("change_in_oi", 0) for opt in calls)
        put_oi_change  = sum(opt.get("change_in_oi", 0) for opt in puts)

        total_call_vol_price = sum(opt.get("volume", 0) * opt.get("last_price", 0) for opt in calls)
        total_put_vol_price  = sum(opt.get("volume", 0) * opt.get("last_price", 0) for opt in puts)
    
        # ----------------------------------------------------
        # 2) Compute PCR
        # ----------------------------------------------------
        pcr_volume    = round(put_volume / max(call_volume, 1), 2)
        pcr_oi        = round(put_oi / max(call_oi, 1), 2)
        pcr_oi_change = round(put_oi_change / max(abs(call_oi_change), 1), 2)
        pcr_vol_price = round(total_put_vol_price / max(total_call_vol_price, 1), 2)

        # ----------------------------------------------------
        # 3) History Storage
        # ----------------------------------------------------
        

        for key in ["oi", "oi_change", "volume"]:
            if key not in hist or not isinstance(hist[key], list):
                hist[key] = []
        
        hist["oi"].append(pcr_oi)
        hist["oi_change"].append(pcr_oi_change)
        hist["volume"].append(pcr_volume) 

        # ----------------------------------------------------
        # 4) SEPARATE SENTIMENT CALCULATIONS
        # ----------------------------------------------------

        if len(hist["oi"]) >= 2:
            prev_oi        = hist["oi"][-2]
            prev_oi_change = hist["oi_change"][-2]
            prev_volume    = hist["volume"][-2]

            # -------------------------------
            # OI SENTIMENT (LATCHED)
            # -------------------------------
            if pcr_oi > prev_oi:
                hist["oi_sentiment"] = "BULLISH"
            elif pcr_oi < prev_oi:
                hist["oi_sentiment"] = "BEARISH"
            # else → KEEP PREVIOUS STATE

            # -------------------------------
            # OI CHANGE SENTIMENT (LATCHED)
            # -------------------------------
            if pcr_oi_change > prev_oi_change:
                hist["oi_change_sentiment"] = "BULLISH"
            elif pcr_oi_change < prev_oi_change:
                hist["oi_change_sentiment"] = "BEARISH"

            # -------------------------------
            # VOLUME SENTIMENT (LATCHED)
            # -------------------------------
            if pcr_volume > prev_volume:
                hist["volume_sentiment"] = "BULLISH"
            elif pcr_volume < prev_volume:
                hist["volume_sentiment"] = "BEARISH"

            # -------------------------------
            # COMBINED SENTIMENT (MAJORITY)
            # -------------------------------
            bullish_count = sum(s == "BULLISH" for s in [
                hist["oi_sentiment"],
                hist["oi_change_sentiment"],
                hist["volume_sentiment"]
            ])

            bearish_count = sum(s == "BEARISH" for s in [
                hist["oi_sentiment"],
                hist["oi_change_sentiment"],
                hist["volume_sentiment"]
            ])

            if bullish_count > bearish_count:
                hist["combined_sentiment"] = "BULLISH"
            elif bearish_count > bullish_count:
                hist["combined_sentiment"] = "BEARISH"
            # else → KEEP PREVIOUS

        # expose for return
        oi_sentiment = hist["oi_sentiment"]
        oi_change_sentiment = hist["oi_change_sentiment"]
        volume_sentiment = hist["volume_sentiment"]
        combined_sentiment = hist["combined_sentiment"]


        # ----------------------------------------------------
        # 5) Initialize Trade Tracking (OI, OI Change, Volume)
        # ----------------------------------------------------
        defaults = {
            "oi_trade": None, "oi_strike": None, "oi_token": None, "oi_entry_price": None,
            "oi_chg_trade": None, "oi_chg_strike": None, "oi_chg_token": None, "oi_chg_entry_price": None,
            "vol_trade": None, "vol_strike": None, "vol_token": None, "vol_entry_price": None
        }

        for k, v in defaults.items():
            if k not in hist:
                hist[k] = v

        # helper to get ATM option
        def get_atm(options):
            if not options:
                return None
            return min(options, key=lambda x: abs(x.get("strike_price", 0) - current_price))

        # helper to get exit price based on stored token/strike
        def get_exit_price(option_type: str, token: Any, strike: Any) -> float:
            options = calls if option_type == "CE" else puts
            # try match by token/ScripCode first
            if token is not None:
                for opt in options:
                    if opt.get("token") == token or opt.get("ScripCode") == token:
                        return float(opt.get("last_price", 0.0))
            # fallback: match by strike
            for opt in options:
                if opt.get("strike_price") == strike:
                    return float(opt.get("last_price", 0.0))
            return 0.0

        # These will be returned for this candle only when exit happens
        oi_exit_price = None
        oi_chg_exit_price = None
        vol_exit_price = None

        # ----------------------------------------------------
        # 6) OI ENTRY / EXIT (Separate)
        # ----------------------------------------------------
        # if hist["oi_trade"] is None:
        #     # ENTRY
        #     if oi_sentiment == "BULLISH":
        #         atm_ce = get_atm(calls)
        #         if atm_ce:
        #             hist["oi_trade"] = "CE"
        #             hist["oi_strike"] = atm_ce.get("strike_price")
        #             hist["oi_token"] = atm_ce.get("token") or atm_ce.get("ScripCode")
        #             hist["oi_entry_price"] = atm_ce.get("last_price", 0.0)

        #     elif oi_sentiment == "BEARISH":
        #         atm_pe = get_atm(puts)
        #         if atm_pe:
        #             hist["oi_trade"] = "PE"
        #             hist["oi_strike"] = atm_pe.get("strike_price")
        #             hist["oi_token"] = atm_pe.get("token") or atm_pe.get("ScripCode")
        #             hist["oi_entry_price"] = atm_pe.get("last_price", 0.0)

        # else:
        #     # EXIT on sentiment flip
        #     if hist["oi_trade"] == "CE" and oi_sentiment == "BEARISH":
        #         oi_exit_price = get_exit_price("CE", hist["oi_token"], hist["oi_strike"])
        #         hist["oi_trade"] = None
        #         hist["oi_strike"] = None
        #         hist["oi_token"] = None
        #         hist["oi_entry_price"] = None

        #     elif hist["oi_trade"] == "PE" and oi_sentiment == "BULLISH":
        #         oi_exit_price = get_exit_price("PE", hist["oi_token"], hist["oi_strike"])
        #         hist["oi_trade"] = None
        #         hist["oi_strike"] = None
        #         hist["oi_token"] = None
        #         hist["oi_entry_price"] = None

        # ----------------------------------------------------
        # 7) OI CHANGE ENTRY / EXIT (Separate)
        # ----------------------------------------------------
        if hist["oi_chg_trade"] is None:
            # ENTRY
            if oi_change_sentiment == "BULLISH":
                atm_ce = get_atm(calls)
                if atm_ce:
                    hist["oi_chg_trade"] = "CE"
                    hist["oi_chg_strike"] = atm_ce.get("strike_price")
                    hist["oi_chg_token"] = atm_ce.get("token") or atm_ce.get("ScripCode")
                    hist["oi_chg_entry_price"] = atm_ce.get("last_price", 0.0)

            elif oi_change_sentiment == "BEARISH":
                atm_pe = get_atm(puts)
                if atm_pe:
                    hist["oi_chg_trade"] = "PE"
                    hist["oi_chg_strike"] = atm_pe.get("strike_price")
                    hist["oi_chg_token"] = atm_pe.get("token") or atm_pe.get("ScripCode")
                    hist["oi_chg_entry_price"] = atm_pe.get("last_price", 0.0)

        else:
            # EXIT on sentiment flip
            if hist["oi_chg_trade"] == "CE" and oi_change_sentiment == "BEARISH":
                oi_chg_exit_price = get_exit_price("CE", hist["oi_chg_token"], hist["oi_chg_strike"])
                hist["oi_chg_trade"] = None
                hist["oi_chg_strike"] = None
                hist["oi_chg_token"] = None
                hist["oi_chg_entry_price"] = None

            elif hist["oi_chg_trade"] == "PE" and oi_change_sentiment == "BULLISH":
                oi_chg_exit_price = get_exit_price("PE", hist["oi_chg_token"], hist["oi_chg_strike"])
                hist["oi_chg_trade"] = None
                hist["oi_chg_strike"] = None
                hist["oi_chg_token"] = None
                hist["oi_chg_entry_price"] = None

        # ----------------------------------------------------
        # 8) VOLUME ENTRY / EXIT (Separate)
        # ----------------------------------------------------
        if hist["vol_trade"] is None:
            # ENTRY
            if volume_sentiment == "BULLISH":
                atm_ce = get_atm(calls)
                if atm_ce:
                    hist["vol_trade"] = "CE"
                    hist["vol_strike"] = atm_ce.get("strike_price")
                    hist["vol_token"] = atm_ce.get("token") or atm_ce.get("ScripCode")
                    hist["vol_entry_price"] = atm_ce.get("last_price", 0.0)

            elif volume_sentiment == "BEARISH":
                atm_pe = get_atm(puts)
                if atm_pe:
                    hist["vol_trade"] = "PE"
                    hist["vol_strike"] = atm_pe.get("strike_price")
                    hist["vol_token"] = atm_pe.get("token") or atm_pe.get("ScripCode")
                    hist["vol_entry_price"] = atm_pe.get("last_price", 0.0)

        else:
            # EXIT on sentiment flip
            if hist["vol_trade"] == "CE" and volume_sentiment == "BEARISH":
                vol_exit_price = get_exit_price("CE", hist["vol_token"], hist["vol_strike"])
                hist["vol_trade"] = None
                hist["vol_strike"] = None
                hist["vol_token"] = None
                hist["vol_entry_price"] = None

            elif hist["vol_trade"] == "PE" and volume_sentiment == "BULLISH":
                vol_exit_price = get_exit_price("PE", hist["vol_token"], hist["vol_strike"])
                hist["vol_trade"] = None
                hist["vol_strike"] = None
                hist["vol_token"] = None
                hist["vol_entry_price"] = None

        
        
       # ----------------------------------------------------
        # 9) Final Return
        # ----------------------------------------------------
        return {
            # raw aggregates
            'call_volume': call_volume,
            'put_volume': put_volume,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'call_oi_change': call_oi_change,
            'put_oi_change': put_oi_change,

            # PCR metrics
            'pcr_volume': pcr_volume,
            'pcr_oi': pcr_oi,
            'pcr_oi_change': pcr_oi_change,
            'pcr_vol_price': pcr_vol_price,

            # sentiments
            'oi_sentiment': oi_sentiment,
            'oi_change_sentiment': oi_change_sentiment,
            'volume_sentiment': volume_sentiment,
            'sentiment': combined_sentiment,

            # OI trade state + prices
            'oi_trade': hist["oi_trade"],                 # CE / PE / None (after this candle)
            'oi_strike': hist["oi_strike"],
            'oi_token': hist["oi_token"],
            'oi_entry_price': hist["oi_entry_price"],
            'oi_exit_price': oi_exit_price,               # only filled on exit candle

            # OI Change trade state + prices
            'oi_change_trade': hist["oi_chg_trade"],
            'oi_change_strike': hist["oi_chg_strike"],
            'oi_change_token': hist["oi_chg_token"],
            'oi_change_entry_price': hist["oi_chg_entry_price"],
            'oi_change_exit_price': oi_chg_exit_price,    # only filled on exit candle

            # Volume trade state + prices
            'volume_trade': hist["vol_trade"],
            'volume_strike': hist["vol_strike"],
            'volume_token': hist["vol_token"],
            'volume_entry_price': hist["vol_entry_price"],
            'volume_exit_price': vol_exit_price,          # only filled on exit candle

            # history for debugging / analysis
            'history': {
                "oi": list(hist["oi"]),
                "oi_change": list(hist["oi_change"]),
                "volume": list(hist["volume"])
            },

            # ---- IV & GREEKS (NEW) ----
            'avg_iv': avg_iv,
            'iv_rank': iv_rank,
            'iv_percentile': iv_percentile,
            'gamma_flip_level': gamma_flip_level,
            'delta_bias_score': delta_bias_score,
            'sl_points': sl_points,
            'target_points': target_points

        }

    def find_dynamic_support_resistance(
                option_chain_data: List[Dict],
                current_price: float
            ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
                """
                Dynamically find top 3 Support and Resistance levels
                using OI, Change in OI, and Volume.
                Works directly with your attribute names.
                """

                # Separate Calls and Puts
                calls = [opt for opt in option_chain_data if opt.get('option_type', '').upper() == 'CE']
                puts = [opt for opt in option_chain_data if opt.get('option_type', '').upper() == 'PE']

                # Weighted strength score (adjust weights if needed)
                def calc_strength(opt: Dict) -> float:
                    oi = opt.get('open_interest', 0)
                    change_in_oi = max(opt.get('change_in_oi', 0), 0)  # ignore unwinding
                    vol = opt.get('volume', 0)
                    return oi * 0.6 + change_in_oi * 0.3 + vol * 0.1

                # 🟢 Supports (Puts below/near current price)
                put_strengths = [
                    (opt.get('strike_price'), calc_strength(opt))
                    for opt in puts if opt.get('strike_price', 0) <= current_price
                ]
                put_strengths.sort(key=lambda x: x[1], reverse=True)
                supports = put_strengths[:3]

                # 🔴 Resistances (Calls above/near current price)
                call_strengths = [
                    (opt.get('strike_price'), calc_strength(opt))
                    for opt in calls if opt.get('strike_price', 0) >= current_price
                ]
                call_strengths.sort(key=lambda x: x[1], reverse=True)
                resistances = call_strengths[:3]

                return supports, resistances
    @staticmethod
    def find_support_resistance_historical(df: pd.DataFrame, current_price: float) -> Tuple[List[float], List[float]]:
                """Find support and resistance levels from historical price data."""
                if df.empty:
                    return [], []
                
                # Find pivot points for support and resistance from historical data
                highs = df['High'].rolling(window=20, center=True).max()
                lows = df['Low'].rolling(window=20, center=True).min()
                
                pivot_highs = df[df['High'] == highs]['High'].unique()
                pivot_lows = df[df['Low'] == lows]['Low'].unique()
                
                # Filter levels with at least 2 touches
                valid_resistances = []
                for h in pivot_highs:
                    touches = ((df['High'] >= h*0.999) & (df['Low'] <= h*1.001)).sum()
                    if touches >= 5 and h > current_price:
                        valid_resistances.append(h)

                valid_supports = []
                for l in pivot_lows:
                    touches = ((df['High'] >= l*0.999) & (df['Low'] <= l*1.001)).sum()
                    if touches >= 5 and l < current_price:
                        valid_supports.append(l)

                # Sort & pick nearest 3
                resistances = sorted(valid_resistances)[:10]
                supports = sorted(valid_supports, reverse=True)[:10]
                # Filter and sort
                # resistances = sorted([h for h in pivot_highs if h > current_price])[:3]
                # supports = sorted([l for l in pivot_lows if l < current_price], reverse=True)[:3]
                
                return supports, resistances
            
    # @staticmethod
    # def find_high_accuracy_support_resistance_combined(
    #             historical_df: pd.DataFrame, 
    #             option_chain_data: List[Dict], 
    #             current_price: float,
    #             tolerance_pct: float = 0.015,  # 1.5% tolerance
    #             min_touch_count: int = 2  # Minimum number of times price touched the level
    #         ) -> Tuple[List[Dict], List[Dict]]:
    #             """
    #             Find high accuracy support and resistance levels by combining:
    #             1. Top 3 option chain OI levels
    #             2. Historical price action validation
    #             3. Multiple touch confirmation
    #             """
    #             if historical_df.empty or not option_chain_data:
    #                 return [], []
                
    #             oc_supports, oc_resistances,market_bias = MarketAnalyzer.find_support_resistance_option_chain(
    #                 option_chain_data, current_price
    #             )
                
    #             # Find swing highs and lows from historical data
    #             swing_highs, swing_lows = MarketAnalyzer.find_swing_points(historical_df)
                
    #             # Create lists of all historical highs and lows for validation
    #             all_highs = historical_df['High'].tolist()
    #             all_lows = historical_df['Low'].tolist()
    #             all_closes = historical_df['Close'].tolist()
                
    #             high_accuracy_supports = []
    #             high_accuracy_resistances = []
                
    #             # Process option chain supports with historical validation
    #             for i, oc_support in enumerate(oc_supports[:3]):  # Top 3 only
    #                 tolerance = oc_support * tolerance_pct
    #                 lower_bound = oc_support - tolerance
    #                 upper_bound = oc_support + tolerance
                    
    #                 # Find historical validation points
    #                 historical_touches = []
    #                 swing_confirmations = []
                    
    #                 # Check swing lows near this level
    #                 for swing_low in swing_lows:
    #                     if lower_bound <= swing_low <= upper_bound:
    #                         swing_confirmations.append(swing_low)
                    
    #                 # Check all lows for touch count
    #                 for low in all_lows:
    #                     if lower_bound <= low <= upper_bound:
    #                         historical_touches.append(low)
                    
    #                 # Check closes for additional validation
    #                 close_touches = [close for close in all_closes 
    #                                 if lower_bound <= close <= upper_bound]
                    
    #                 total_touches = len(historical_touches) + len(close_touches)
                    
    #                 # Calculate accuracy score based on multiple factors
    #                 accuracy_score = 0
                    
    #                 # Factor 1: Number of swing confirmations (high weight)
    #                 accuracy_score += len(swing_confirmations) * 30
                    
    #                 # Factor 2: Total touch count (medium weight)
    #                 if total_touches >= min_touch_count:
    #                     accuracy_score += min(total_touches * 15, 60)  # Cap at 60
                    
    #                 # Factor 3: Recent validation (check last 20 candles)
    #                 recent_data = historical_df.tail(20)
    #                 recent_validation = any(
    #                     lower_bound <= low <= upper_bound 
    #                     for low in recent_data['Low'].tolist()
    #                 )
    #                 if recent_validation:
    #                     accuracy_score += 25
                    
    #                 # Only include if meets minimum criteria
    #                 if accuracy_score >= 50 and total_touches >= min_touch_count:
    #                     # Use the most frequent level or average of swing confirmations
    #                     if swing_confirmations:
    #                         validated_level = sum(swing_confirmations) / len(swing_confirmations)
    #                     else:
    #                         validated_level = oc_support
                        
    #                     high_accuracy_supports.append({
    #                         'level': round(validated_level, 2),
    #                         'oi_support': oc_support,
    #                         'accuracy_score': accuracy_score,
    #                         'touch_count': total_touches,
    #                         'swing_confirmations': len(swing_confirmations),
    #                         'strength': 'HIGH' if accuracy_score >= 80 else 'MEDIUM',
    #                         'rank': i + 1
    #                     })
                
    #             # Process option chain resistances with historical validation
    #             for i, oc_resistance in enumerate(oc_resistances[:3]):  # Top 3 only
    #                 tolerance = oc_resistance * tolerance_pct
    #                 lower_bound = oc_resistance - tolerance
    #                 upper_bound = oc_resistance + tolerance
                    
    #                 # Find historical validation points
    #                 historical_touches = []
    #                 swing_confirmations = []
                    
    #                 # Check swing highs near this level
    #                 for swing_high in swing_highs:
    #                     if lower_bound <= swing_high <= upper_bound:
    #                         swing_confirmations.append(swing_high)
                    
    #                 # Check all highs for touch count
    #                 for high in all_highs:
    #                     if lower_bound <= high <= upper_bound:
    #                         historical_touches.append(high)
                    
    #                 # Check closes for additional validation
    #                 close_touches = [close for close in all_closes 
    #                                 if lower_bound <= close <= upper_bound]
                    
    #                 total_touches = len(historical_touches) + len(close_touches)
                    
    #                 # Calculate accuracy score
    #                 accuracy_score = 0
    #                 accuracy_score += len(swing_confirmations) * 30
                    
    #                 if total_touches >= min_touch_count:
    #                     accuracy_score += min(total_touches * 15, 60)
                    
    #                 # Recent validation
    #                 recent_data = historical_df.tail(20)
    #                 recent_validation = any(
    #                     lower_bound <= high <= upper_bound 
    #                     for high in recent_data['High'].tolist()
    #                 )
    #                 if recent_validation:
    #                     accuracy_score += 25
                    
    #                 if accuracy_score >= 50 and total_touches >= min_touch_count:
    #                     if swing_confirmations:
    #                         validated_level = sum(swing_confirmations) / len(swing_confirmations)
    #                     else:
    #                         validated_level = oc_resistance
                        
    #                     high_accuracy_resistances.append({
    #                         'level': round(validated_level, 2),
    #                         'oi_resistance': oc_resistance,
    #                         'accuracy_score': accuracy_score,
    #                         'touch_count': total_touches,
    #                         'swing_confirmations': len(swing_confirmations),
    #                         'strength': 'HIGH' if accuracy_score >= 80 else 'MEDIUM',
    #                         'rank': i + 1
    #                     })
                
    #             # Sort by accuracy score (highest first)
    #             high_accuracy_supports.sort(key=lambda x: x['accuracy_score'], reverse=True)
    #             high_accuracy_resistances.sort(key=lambda x: x['accuracy_score'], reverse=True)
                
    #             return high_accuracy_supports[:10], high_accuracy_resistances[:10]
            
    @staticmethod
    def find_high_accuracy_support_resistance_combined_latest(
                historical_df: pd.DataFrame, 
                option_chain_data: List[Dict], 
                current_price: float,
                pclose_price: float,
                symbol: str,
                tolerance_pct: float = 0.015,
                min_touch_count: int = 2
            ) -> Tuple[List[Dict], List[Dict]]:

        # STEP 0: COMPUTE OPTION ANALYSIS INSIDE
        option_analysis = MarketAnalyzer.analyze_option_chain(option_chain_data,current_price,pclose_price,symbol)
        # print(option_analysis)
        market_bias = option_analysis.get("sentiment", "NEUTRAL")

        # Step 1: Historical S/R
        hist_supports, hist_resistances = MarketAnalyzer.find_support_resistance_historical(
            historical_df, current_price
        )

        # Step 2: Swings
        swing_highs, swing_lows = MarketAnalyzer.find_swing_points(historical_df)

        # Step 3: Option Chain Levels (S/R from OI)
        oc_supports, oc_resistances, _ = MarketAnalyzer.find_support_resistance_option_chain(
            option_chain_data, current_price, pclose_price
        )

        tolerance_points = 30

        # Step 4: Validate Supports
        validated_supports = []
        for oc_support in oc_supports:
            lower_limit = oc_support - tolerance_points
            upper_limit = oc_support + tolerance_points

            valid_hist = [s for s in hist_supports if lower_limit <= s <= upper_limit]
            valid_swing = [l for l in swing_lows if lower_limit <= l <= upper_limit]

            if valid_hist or valid_swing:
                validated_supports.append({
                    'level': oc_support,
                    'historical_matches': valid_hist,
                    'swing_matches': valid_swing,
                    'type': 'SUPPORT'
                })

        # Step 4: Validate Resistances
        validated_resistances = []
        for oc_resistance in oc_resistances:
            lower_limit = oc_resistance - tolerance_points
            upper_limit = oc_resistance + tolerance_points

            valid_hist = [r for r in hist_resistances if lower_limit <= r <= upper_limit]
            valid_swing = [h for h in swing_highs if lower_limit <= h <= upper_limit]
            if valid_hist or valid_swing:
                validated_resistances.append({
                    'level': oc_resistance,
                    'historical_matches': valid_hist,
                    'swing_matches': valid_swing,
                    'type': 'RESISTANCE'
                })

        # Step 5: Nearest call/put
        call_options = [opt for opt in option_chain_data if opt['option_type'] == 'CE']
        put_options  = [opt for opt in option_chain_data if opt['option_type'] == 'PE']

        nearest_call = min(call_options, key=lambda x: abs(x['strike_price'] - current_price)) if call_options else None
        nearest_put  = min(put_options,  key=lambda x: abs(x['strike_price'] - current_price)) if put_options  else None
        nearest_call_info = {
            'name': nearest_call['name'],
            'scripCode': nearest_call['ScripCode'],
            'strike_price': nearest_call['strike_price'],
            'last_price': nearest_call['last_price']
        } if nearest_call else None

        nearest_put_info = {
            'name': nearest_put['name'],
            'scripCode': nearest_put['ScripCode'],
            'strike_price': nearest_put['strike_price'],
            'last_price': nearest_put['last_price']
        } if nearest_put else None

        # Final Output including the PCR analysis
        return {
            'current_price': current_price,
            'nearest_call': nearest_call_info,
            'nearest_put': nearest_put_info,
            'validated_supports': validated_supports,
            'validated_resistances': validated_resistances,
            'market_bias': market_bias,
            'oc_supports': oc_supports,
            'oc_resistances': oc_resistances,

            # NEW FIELD RETURNED
            'option_analysis': option_analysis
        }


    @staticmethod
    def find_swing_points(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
                """Find swing highs and lows."""
                if df.empty or len(df) < 5:
                    return [], []
                
                swing_highs = []
                swing_lows = []
                
                for i in range(2, len(df) - 2):
                    # Swing High: higher than 2 periods before and after
                    if (df.iloc[i]['High'] > df.iloc[i-2]['High'] and 
                        df.iloc[i]['High'] > df.iloc[i-1]['High'] and
                        df.iloc[i]['High'] > df.iloc[i+1]['High'] and
                        df.iloc[i]['High'] > df.iloc[i+2]['High']):
                        swing_highs.append(df.iloc[i]['High'])
                    
                    # Swing Low: lower than 2 periods before and after
                    if (df.iloc[i]['Low'] < df.iloc[i-2]['Low'] and 
                        df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and
                        df.iloc[i]['Low'] < df.iloc[i+1]['Low'] and
                        df.iloc[i]['Low'] < df.iloc[i+2]['Low']):
                        swing_lows.append(df.iloc[i]['Low'])
                
                return swing_highs, swing_lows
            
    @staticmethod
    def find_supply_demand_zones(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
                """Find supply and demand zones."""
                swing_highs, swing_lows = MarketAnalyzer.find_swing_points(df)
                
                # Supply zones around swing highs
                supply_zones = []
                for high in swing_highs[-5:]:  # Last 5 swing highs
                    supply_zones.append({
                        'level': high,
                        'zone_top': high + (high * 0.002),  # 0.2% above
                        'zone_bottom': high - (high * 0.002),  # 0.2% below
                        'strength': 'MEDIUM'
                    })
                
                # Demand zones around swing lows
                demand_zones = []
                for low in swing_lows[-5:]:  # Last 5 swing lows
                    demand_zones.append({
                        'level': low,
                        'zone_top': low + (low * 0.002),  # 0.2% above
                        'zone_bottom': low - (low * 0.002),  # 0.2% below
                        'strength': 'MEDIUM'
                    })
                
                return supply_zones, demand_zones
            
    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame) -> Dict[str, float]:
                """Calculate Fibonacci retracement levels."""
                if df.empty:
                    return {}
                
                high = df['High'].max()
                low = df['Low'].min()
                diff = high - low
                
                fib_levels = {
                    'Fib_0': high,
                    'Fib_23.6': high - 0.236 * diff,
                    'Fib_38.2': high - 0.382 * diff,
                    'Fib_50': high - 0.5 * diff,
                    'Fib_61.8': high - 0.618 * diff,
                    'Fib_78.6': high - 0.786 * diff,
                    'Fib_100': low
                }
                
                return fib_levels
            
    @staticmethod
    def calculate_fibonacci_extension_levels(df: pd.DataFrame) -> Dict[str, float]:
                """Calculate Fibonacci extension levels for projecting targets."""
                if df.empty:
                    return {}
                
                high = df['High'].max()
                low = df['Low'].min()
                diff = high - low
                
                # Extensions beyond the range (for breakout targets)
                extension_levels = {
                    'Ext_127.2': high + 0.272 * diff,
                    'Ext_138.2': high + 0.382 * diff,
                    'Ext_161.8': high + 0.618 * diff,
                    'Ext_200': high + diff,
                    'Ext_261.8': high + 1.618 * diff,
                    # Downside extensions
                    'Ext_Down_127.2': low - 0.272 * diff,
                    'Ext_Down_138.2': low - 0.382 * diff,
                    'Ext_Down_161.8': low - 0.618 * diff,
                    'Ext_Down_200': low - diff,
                    'Ext_Down_261.8': low - 1.618 * diff
                }
                
                return extension_levels
            
    @staticmethod
    def analyze_fibonacci_trend(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
                """
                Enhanced Fibonacci trend analysis with accurate uptrend and downtrend detection.
                Analyzes price action relative to Fibonacci levels and recent swing movements.
                """
                if df.empty or len(df) < 50:
                    return {'trend': 'INSUFFICIENT_DATA', 'strength': 'WEAK', 'key_levels': []}
                
                # Calculate Fibonacci levels from recent swing high to swing low
                recent_data = df.tail(100)  # Last 100 periods for trend analysis
                
                # Find recent significant swing high and low
                swing_high_idx = recent_data['High'].idxmax()
                swing_low_idx = recent_data['Low'].idxmin()
                
                swing_high = recent_data['High'].max()
                swing_low = recent_data['Low'].min()
                
                # Determine if we're in uptrend or downtrend scenario based on swing sequence
                is_uptrend_scenario = swing_low_idx < swing_high_idx  # Low came before high
                
                # Calculate Fibonacci levels
                diff = swing_high - swing_low
                
                if is_uptrend_scenario:
                    # Uptrend Fibonacci retracement (from high back to low)
                    fib_levels = {
                        'fib_0': swing_high,
                        'fib_23.6': swing_high - 0.236 * diff,
                        'fib_38.2': swing_high - 0.382 * diff,
                        'fib_50': swing_high - 0.5 * diff,
                        'fib_61.8': swing_high - 0.618 * diff,
                        'fib_78.6': swing_high - 0.786 * diff,
                        'fib_100': swing_low
                    }
                else:
                    # Downtrend Fibonacci retracement (from low back up to high)
                    fib_levels = {
                        'fib_0': swing_low,
                        'fib_23.6': swing_low + 0.236 * diff,
                        'fib_38.2': swing_low + 0.382 * diff,
                        'fib_50': swing_low + 0.5 * diff,
                        'fib_61.8': swing_low + 0.618 * diff,
                        'fib_78.6': swing_low + 0.786 * diff,
                        'fib_100': swing_high
                    }
                
                # Analyze recent price action (last 20 periods)
                recent_prices = df['Close'].tail(20)
                recent_highs = df['High'].tail(20)
                recent_lows = df['Low'].tail(20)
                
                price_momentum = (current_price - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
                volatility = recent_prices.std() / recent_prices.mean() * 100
                
                # Trend determination logic
                trend_analysis = MarketAnalyzer._determine_fibonacci_trend(
                    current_price, fib_levels, is_uptrend_scenario, recent_prices, 
                    recent_highs, recent_lows, price_momentum, swing_high, swing_low
                )
                
                # Calculate extension levels for targets
                extension_levels = MarketAnalyzer._calculate_trend_extensions(
                    swing_high, swing_low, diff, trend_analysis['trend']
                )
                
                return {
                    'trend': trend_analysis['trend'],
                    'strength': trend_analysis['strength'],
                    'scenario': 'UPTREND_RETRACEMENT' if is_uptrend_scenario else 'DOWNTREND_RETRACEMENT',
                    'current_fib_zone': trend_analysis['current_zone'],
                    'key_levels': trend_analysis['key_levels'],
                    'extensions': extension_levels,
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'momentum': round(price_momentum, 2),
                    'volatility': round(volatility, 2),
                    'fib_support': trend_analysis.get('fib_support', 0),
                    'fib_resistance': trend_analysis.get('fib_resistance', 0),
                    'signals': trend_analysis.get('signals', [])
                }
            
    @staticmethod
    def _determine_fibonacci_trend(current_price: float, fib_levels: Dict, is_uptrend_scenario: bool,
                                        recent_prices: pd.Series, recent_highs: pd.Series, recent_lows: pd.Series,
                                        price_momentum: float, swing_high: float, swing_low: float) -> Dict[str, Any]:
                """Determine trend based on Fibonacci analysis."""
                
                fib_23 = fib_levels['fib_23.6']
                fib_38 = fib_levels['fib_38.2']
                fib_50 = fib_levels['fib_50']
                fib_61 = fib_levels['fib_61.8']
                fib_78 = fib_levels['fib_78.6']
                
                signals = []
                current_zone = "UNKNOWN"
                key_levels = []
                
                if is_uptrend_scenario:
                    # Uptrend retracement analysis
                    if current_price > fib_23:
                        current_zone = "ABOVE_23.6"
                        trend = "STRONG_UPTREND"
                        strength = "VERY_STRONG"
                        key_levels = [fib_23, fib_38]
                        fib_support = fib_23
                        fib_resistance = swing_high
                        signals.append("Price holding above 23.6% - Very bullish")
                        
                    elif current_price > fib_38:
                        current_zone = "ABOVE_38.2"
                        # Check if price bounced from fib levels
                        recent_low = recent_lows.min()
                        if recent_low <= fib_38 and current_price > fib_38:
                            trend = "UPTREND"
                            strength = "STRONG"
                            signals.append("Bounce from 38.2% fib - Bullish continuation")
                        else:
                            trend = "UPTREND"
                            strength = "MODERATE"
                        key_levels = [fib_38, fib_50]
                        fib_support = fib_38
                        fib_resistance = swing_high
                        
                    elif current_price > fib_50:
                        current_zone = "ABOVE_50"
                        recent_low = recent_lows.min()
                        if recent_low <= fib_50 and current_price > fib_50 and price_momentum > 0:
                            trend = "UPTREND"
                            strength = "MODERATE"
                            signals.append("Holding above 50% after retest - Cautiously bullish")
                        else:
                            trend = "CONSOLIDATING"
                            strength = "WEAK"
                        key_levels = [fib_50, fib_61]
                        fib_support = fib_50
                        fib_resistance = fib_38
                        
                    elif current_price > fib_61:
                        current_zone = "ABOVE_61.8"
                        trend = "WEAK_UPTREND"
                        strength = "WEAK"
                        key_levels = [fib_61, fib_78]
                        fib_support = fib_61
                        fib_resistance = fib_50
                        signals.append("Deep retracement - Trend weakening")
                        
                    elif current_price > fib_78:
                        current_zone = "ABOVE_78.6"
                        trend = "TREND_REVERSAL_RISK"
                        strength = "VERY_WEAK"
                        key_levels = [fib_78, swing_low]
                        fib_support = fib_78
                        fib_resistance = fib_61
                        signals.append("Very deep retracement - High reversal risk")
                        
                    else:
                        current_zone = "BELOW_78.6"
                        trend = "DOWNTREND"
                        strength = "STRONG"
                        key_levels = [swing_low, fib_78]
                        fib_support = swing_low
                        fib_resistance = fib_78
                        signals.append("Break below 78.6% - Trend reversal confirmed")
                
                else:
                    # Downtrend retracement analysis
                    if current_price < fib_23:
                        current_zone = "BELOW_23.6"
                        trend = "STRONG_DOWNTREND"
                        strength = "VERY_STRONG"
                        key_levels = [fib_23, fib_38]
                        fib_resistance = fib_23
                        fib_support = swing_low
                        signals.append("Price holding below 23.6% - Very bearish")
                        
                    elif current_price < fib_38:
                        current_zone = "BELOW_38.2"
                        recent_high = recent_highs.max()
                        if recent_high >= fib_38 and current_price < fib_38:
                            trend = "DOWNTREND"
                            strength = "STRONG"
                            signals.append("Rejection at 38.2% fib - Bearish continuation")
                        else:
                            trend = "DOWNTREND"
                            strength = "MODERATE"
                        key_levels = [fib_38, fib_50]
                        fib_resistance = fib_38
                        fib_support = swing_low
                        
                    elif current_price < fib_50:
                        current_zone = "BELOW_50"
                        recent_high = recent_highs.max()
                        if recent_high >= fib_50 and current_price < fib_50 and price_momentum < 0:
                            trend = "DOWNTREND"
                            strength = "MODERATE"
                            signals.append("Rejection at 50% - Cautiously bearish")
                        else:
                            trend = "CONSOLIDATING"
                            strength = "WEAK"
                        key_levels = [fib_50, fib_61]
                        fib_resistance = fib_50
                        fib_support = fib_38
                        
                    elif current_price < fib_61:
                        current_zone = "BELOW_61.8"
                        trend = "WEAK_DOWNTREND"
                        strength = "WEAK"
                        key_levels = [fib_61, fib_78]
                        fib_resistance = fib_61
                        fib_support = fib_50
                        signals.append("Deep retracement - Trend weakening")
                        
                    elif current_price < fib_78:
                        current_zone = "BELOW_78.6"
                        trend = "TREND_REVERSAL_RISK"
                        strength = "VERY_WEAK"
                        key_levels = [fib_78, swing_high]
                        fib_resistance = fib_78
                        fib_support = fib_61
                        signals.append("Very deep retracement - High reversal risk")
                        
                    else:
                        current_zone = "ABOVE_78.6"
                        trend = "UPTREND"
                        strength = "STRONG"
                        key_levels = [swing_high, fib_78]
                        fib_resistance = swing_high
                        fib_support = fib_78
                        signals.append("Break above 78.6% - Trend reversal confirmed")
                
                return {
                    'trend': trend,
                    'strength': strength,
                    'current_zone': current_zone,
                    'key_levels': key_levels,
                    'fib_support': fib_support,
                    'fib_resistance': fib_resistance,
                    'signals': signals
                }
            
    @staticmethod
    def _calculate_trend_extensions(swing_high: float, swing_low: float, diff: float, trend: str) -> Dict[str, float]:
                """Calculate Fibonacci extension levels for trend targets."""
                extensions = {}
                
                if 'UPTREND' in trend:
                    extensions = {
                        'target_127.2': swing_high + 0.272 * diff,
                        'target_138.2': swing_high + 0.382 * diff,
                        'target_161.8': swing_high + 0.618 * diff,
                        'target_200': swing_high + diff
                    }
                elif 'DOWNTREND' in trend:
                    extensions = {
                        'target_127.2': swing_low - 0.272 * diff,
                        'target_138.2': swing_low - 0.382 * diff,
                        'target_161.8': swing_low - 0.618 * diff,
                        'target_200': swing_low - diff
                    }
                
                return extensions

# ===============================
# MAIN TRADING SYSTEM
# ===============================

class SimplifiedTradingSystem:
    """Professional trading system focusing on analysis."""
    
    def __init__(self, credentials_path: str = "credentials.json"):
        self.db_manager = DatabaseManager()
        self.api_client = APIClient(credentials_path)
        self.analyzer = MarketAnalyzer()
         # -----------------------------------
        # DELTA + OI TRADE STATE (SEPARATE)
        # -----------------------------------
        self.delta_oi_state = {
            "active": False,
            "side": None,          # BUY_CE / BUY_PE
            "entry_price": None
        }
        # -----------------------------------
        # Initialize API client
        if not self.api_client.initialize_client():
            raise Exception("Failed to initialize API client")
        
    
    def display_positions(self) -> None:
        """Display current positions with professional formatting."""
        positions = self.api_client.get_positions()
        
        if not positions:
            print("\nACTIVE POSITIONS: None")
            return
        
        print(f"\n{'='*110}")
        print(f"ACTIVE POSITIONS ({len(positions)} positions) - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*110}")
        print(f"{'#':<3} {'ScripName':<40} {'Code':<8} {'AvgRate':<9} {'LTP':<9} {'NetQty':<8} {'P&L':<12} {'Day%':<8}")
        print(f"{'='*110}")
        
        total_pnl = 0.0
        total_invested = 0.0
        
        for i, pos in enumerate(positions, 1):
            scrip_name = pos.get('ScripName', 'Unknown')[:38]  # Truncate long names
            scrip_code = pos.get('ScripCode', 0)
            avg_rate = pos.get('AvgRate', 0)
            ltp = pos.get('LTP', 0)
            net_qty = pos.get('NetQty', 0)
            mtom = pos.get('MTOM', 0)  # Mark to Market P&L
            prev_close = pos.get('PreviousClose', 0)
            
            # Calculate day change percentage
            day_change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
            
            # Calculate investment value
            investment = abs(avg_rate * net_qty)
            total_invested += investment
            total_pnl += mtom
            
            # Color coding for P&L (using text indicators)
            pnl_indicator = "+" if mtom >= 0 else "-"
            day_indicator = "+" if day_change_pct >= 0 else ""
            
            print(f"{i:<3} {scrip_name:<40} {scrip_code:<8} "
                  f"Rs.{avg_rate:<6.2f} Rs.{ltp:<6.2f} {net_qty:<8} "
                  f"{pnl_indicator}Rs.{abs(mtom):<9.2f} {day_indicator}{day_change_pct:<6.2f}%")
        
        print(f"{'='*110}")
        print(f"TOTAL INVESTMENT: Rs.{total_invested:,.2f}")
        print(f"TOTAL P&L: Rs.{total_pnl:,.2f} ({(total_pnl/total_invested*100) if total_invested else 0:.2f}%)")
        print(f"{'='*110}")
    
    def display_positions_compact(self) -> None:
        """Display positions in compact format."""
        positions = self.api_client.get_positions()
        
        if not positions:
            print("No active positions")
            return
        
        total_pnl = sum(pos.get('MTOM', 0) for pos in positions)
        
        print("\nACTIVE POSITIONS SUMMARY:")
        print("-" * 80)
        print(f"{'Scrip':40} {'Qty':>6} {'LTP':>10} {'P&L':>12}")
        print("-" * 80)
        for pos in positions:
            scrip = pos.get('ScripName', '')[:40]
            qty = pos.get('NetQty', 0)
            ltp = pos.get('LTP', 0)
            pnl = pos.get('MTOM', 0)
            print(f"{scrip:40} {qty:6} {ltp:10.2f} {pnl:12.2f}")
        
        print("-" * 80)
        print(f"{'Total P&L':>58}: {total_pnl:12.2f}")
        print("-" * 80)
    
    def run_analysis_loop(self) -> None:
        """Run the main analysis loop."""

        try:
            cycle_count = 0
                        

            while True:
                # ----------------------------------
                # 1️⃣ CHECK MARKET STATUS INSIDE LOOP
                # ----------------------------------
                # market_status = self.api_client.get_market_status()

                # if not market_status.get("is_open"):
                #     Logger.warning(
                #         f"{market_status.get('message', 'Market closed')} | Waiting..."
                #     )  
                #     time.sleep(30) 
                #     continue  # ✅ NOW THIS IS LEGAL
                if self.api_client.is_trading_time():
                    cycle_count += 1
                    current_time = datetime.now()
                    
                    print(f"\n{'='*60}")
                    print(f"ANALYSIS CYCLE #{cycle_count} - {current_time.strftime('%H:%M:%S')}")
                    print(f"{'='*60}")
                    
                    # Analyze each symbol
                    for symbol in CONFIG.SYMBOLS:
                        try:
                            self.analyze_symbol(symbol)
                        except Exception as e:
                            Logger.error(f"Error analyzing {symbol}: {e}")
                    
                    # Display positions (choose compact or full display)
                    # self.display_positions_compact()  # or use self.display_positions() for full display
                    
                    # Wait before next cycle
                    Logger.info(f"Cycle complete. Waiting {CONFIG.DATA_UPDATE_INTERVAL} seconds...")
                    time.sleep(CONFIG.DATA_UPDATE_INTERVAL)
                else:
                    Logger.warning("Market is closed. Waiting for next trading session...")
                    time.sleep(30)
        except KeyboardInterrupt:
            Logger.info("Trading system stopped by user")
        except Exception as e:
            Logger.error(f"Critical system error: {e}")
    
    def analyze_symbol(self, symbol: str) -> None:
        """Complete symbol analysis with enhanced support/resistance detection."""
        # 1. Load option chain data and store
        option_chain = self.api_client.get_option_chain_data(symbol)
        if option_chain:
            self.db_manager.store_option_chain_data(symbol, option_chain)
        
        # 2. Load symbol feed
        current_data = self.api_client.get_current_price(symbol)
        if not current_data:
            Logger.error(f"No current price data for {symbol}")
            return
        
        current_price = current_data['ltp']
        open_price = current_data.get('open', 0)
        pclose_price = current_data.get('pclose', 0)
        # Store market data
        # self.db_manager.store_market_data(symbol, current_data)
        
        # 3. Load historical data (100 days, 5-minute data)
        historical_data = self.api_client.fetch_historical_data(symbol, mins="5m", days_back=100)
        
        # historical_data15 = self.api_client.fetch_historical_data(symbol, mins="15m", days_back=100)

        if historical_data is None or historical_data.empty:
            Logger.error(f"No historical data for {symbol}")
            return
        
        # 4. Analyze option chain
        # option_analysis = self.analyzer.analyze_option_chain(option_chain)
        
        # 5. Find support and resistance from OPTION CHAIN
        # oc_supports, oc_resistances,market_bias = self.analyzer.find_support_resistance_option_chain(
        #     option_chain, current_price, pclose_price
        # )
        
        # # 6. Find support and resistance from HISTORICAL DATA
        # hist_supports, hist_resistances = self.analyzer.find_support_resistance_historical(
        #     historical_data, current_price
        # )
        # hist_supports15, hist_resistances15 = self.analyzer.find_support_resistance_historical(
        #     historical_data15, current_price
        # )
        
        # 7. NEW: Enhanced high accuracy support and resistance detection
        # high_accuracy_supports, high_accuracy_resistances = self.analyzer.find_high_accuracy_support_resistance_combined(
        #     historical_data, option_chain, current_price
        # )
        
        
        
        # # Display enhanced comprehensive analysis
        # self.display_enhanced_analysis(
        #     symbol, current_price, current_data, option_analysis, 
        #     oc_supports, oc_resistances, 
        #     high_accuracy_supports, high_accuracy_resistances
        # )
        result = self.analyzer.find_high_accuracy_support_resistance_combined_latest(historical_data, option_chain, current_price,open_price,symbol)
        option_analysis = result["option_analysis"]

        # oi_sentiment = option_analysis.get("oi_sentiment")
        # delta_score = option_analysis.get("delta_bias_score", 50)
        # gamma_flip = option_analysis.get("gamma_flip_level", 0)
        # pcr_oi = option_analysis.get("pcr_oi", 0)
        # price = current_price

        current_price = result['current_price']
        nearest_call = result['nearest_call']
        nearest_put = result['nearest_put']
        validated_supports = result['validated_supports']
        validated_resistances = result['validated_resistances']
        oc_supports = result['oc_supports']
        oc_resistances = result['oc_resistances']
        market_bias = result['market_bias']

        hist_df = TechnicalIndicators.compute_all_indicators(historical_data)

        # Now call the new display function
        self.display_enhanced_analysis_latest(
            symbol,
            current_price,
            nearest_call,
            nearest_put,
            current_data,
            option_analysis,
            oc_supports,
            oc_resistances,
            market_bias,
            validated_supports,
            validated_resistances,
            hist_df,
            option_chain
        )

    def display_enhanced_analysis_latest(
        self,
        symbol,
        current_price,
        nearest_call,
        nearest_put,
        current_data,
        option_analysis,
        oc_supports,
        oc_resistances,
        market_bias,
        validated_supports,
        validated_resistances,
        hist_df,
        option_chain
    ):
        # ==================================================
        # STEP 1️⃣ MARKET CONTEXT
        # ==================================================
        last = hist_df.iloc[-1]
        ema9 = round(last.get("EMA_9", 0), 2)
        ema21 = round(last.get("EMA_21", 0), 2)

        oi_sentiment = option_analysis.get("oi_sentiment", "NEUTRAL")
        delta_bias = option_analysis.get("delta_bias_score", 50)
        pcr_oi = option_analysis.get("pcr_oi", 1.0)

        print(f"\n{'=' * 80}")
        print(f"MARKET ANALYSIS: {symbol}")
        print(f"{'=' * 80}")
        print(f"Price: Rs.{current_price:.2f}")
        print(
            f"O:{current_data.get('open',0):.2f} | "
            f"H:{current_data.get('high',0):.2f} | "
            f"L:{current_data.get('low',0):.2f} | "
            f"Vol:{current_data.get('volume',0):,}"
        )
        # print(f"EMA9: {ema9} | EMA21: {ema21}")
        # print(f"Delta Bias: {delta_bias} | PCR-OI: {pcr_oi}")
 

        # ==================================================
        # STEP 2️⃣ STATE INIT
        # ==================================================
        ALLOWED_S5_SYMBOLS = {"NIFTY"}

        if symbol not in CONFIG.oi_state:
            CONFIG.oi_state[symbol] = {
                "bullish": False,
                "bearish": False,
                "bullish_level": None,
                "bearish_level": None,
                "trades": {
                    s: {"CE": new_trade_state(), "PE": new_trade_state()}
                    for s in ["S1", "S2", "S3", "S4", "S5"]
                }
            }

        state = CONFIG.oi_state[symbol]

        # ==================================================
        # STEP 3️⃣ PRICE CONFIRMATION
        # ==================================================
        candle_low = current_data["low"]
        candle_high = current_data["high"]

        state["bullish"] = state["bearish"] = False

        supports = {x for s in validated_supports for x in s.get("historical_matches", [])}
        resistances = {x for r in validated_resistances for x in r.get("historical_matches", [])}
       

        for lvl in supports:
            if candle_low < lvl < current_price:
                state["bullish"] = True
                state["bullish_level"] = lvl

        for lvl in resistances:
            if candle_high > lvl > current_price:
                state["bearish"] = True
                state["bearish_level"] = lvl
        # print(resistances)
        # ==================================================
        # STEP 4️⃣ STRATEGY ENTRY CONDITIONS
        # ==================================================
        strategies = {
            "S1": lambda: True,
            "S2": lambda: delta_bias > 50,
            "S3": lambda: ema9 > ema21,
            "S4": lambda: (
                state["bullish"] and
                delta_bias > 50 and
                ema9 > ema21
            ),            
            "S5": lambda: True
        }

        # ==================================================
        # STEP 5️⃣ ENTRY (ORDERS + DB)
        # ==================================================
        for sid, cond in strategies.items():

            # ---------- CE ENTRY ----------
            ce = state["trades"][sid]["CE"]
            if (
                not ce["active"]
                and nearest_call
                and oi_sentiment == "BULLISH"
                and state["bullish"]
                and cond()
                and state["bullish_level"] != ce["last_used_level"]
                and self.api_client.is_live_entry_time()
            ):
                qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots
                price = nearest_call["last_price"]

                trade_id = self.db_manager.insert_trade(
                    symbol=symbol,
                    strategy=sid,
                    option_type="CE",
                    strike=nearest_call["strike_price"],
                    token=nearest_call["scripCode"],
                    qty=qty,
                    entry_price=price,
                    entry_oi=oi_sentiment,
                    entry_delta=delta_bias
                )

                ce.update({
                    "active": True,
                    "trade_id": trade_id,
                    "strike": nearest_call["strike_price"],
                    "token": nearest_call["scripCode"],
                    "entry_price": price,
                    "qty": qty,
                    "entry_oi": oi_sentiment,
                    "entry_delta": delta_bias,
                    "entry_pcr_oi": pcr_oi,
                    "last_used_level": state["bullish_level"]
                })

                if sid == "S5" and symbol.upper() in ALLOWED_S5_SYMBOLS:
                    self.api_client.place_order_api(
                        scripCode=ce["token"],
                        direction="BUY",
                        quantity=qty,
                        price=price,
                        strike_price=ce["strike"],
                        option_type="CE",
                        exchange=current_data.get("Exch", "N")
                    )

            # ---------- PE ENTRY ----------
            pe = state["trades"][sid]["PE"]
            bearish_ok = (
                sid == "S1" or
                (sid == "S2" and delta_bias < 50) or
                (sid == "S3" and ema9 < ema21) or
                (sid == "S4" and state["bearish"] and delta_bias < 50 and ema9 < ema21) or
                sid == "S5"
            )

            if (
                not pe["active"]
                and nearest_put
                and oi_sentiment == "BEARISH"
                and state["bearish"]
                and bearish_ok
                and state["bearish_level"] != pe["last_used_level"]
                and self.api_client.is_live_entry_time()
            ):
                qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots
                price = nearest_put["last_price"]

                trade_id = self.db_manager.insert_trade(
                    symbol=symbol,
                    strategy=sid,
                    option_type="PE",
                    strike=nearest_put["strike_price"],
                    token=nearest_put["scripCode"],
                    qty=qty,
                    entry_price=price,
                    entry_oi=oi_sentiment,
                    entry_delta=delta_bias
                )

                pe.update({
                    "active": True,
                    "trade_id": trade_id,
                    "strike": nearest_put["strike_price"],
                    "token": nearest_put["scripCode"],
                    "entry_price": price,
                    "qty": qty,  
                    "entry_oi": oi_sentiment,
                    "entry_delta": delta_bias,
                    "entry_pcr_oi": pcr_oi,
                    "last_used_level": state["bearish_level"]
                })


                if sid == "S5" and symbol.upper() in ALLOWED_S5_SYMBOLS:
                    self.api_client.place_order_api(
                        scripCode=pe["token"],
                        direction="BUY",
                        quantity=qty,
                        price=price,
                        strike_price=pe["strike"],
                        option_type="PE",
                        exchange=current_data.get("Exch", "N")
                    )

        # ==================================================
        # STEP 6️⃣ EXIT + P&L
        # ==================================================
        for sid, sides in state["trades"].items():
            for opt_type, trade in sides.items():

                if not trade["active"]:
                    continue

                exit_now = False
                exit_reason = None

                def get_exit_price(option_type, token, strike):
                    for opt in option_chain:
                        if opt.get("option_type") == option_type and (
                            opt.get("ScripCode") == token or opt.get("strike_price") == strike
                        ):
                            return float(opt.get("last_price", 0))
                    return 0.0
                


                # ---------- S1 (PCR-OI) ----------
                if sid == "S1":
                    if opt_type == "CE":
                        exit_now = pcr_oi < trade["entry_pcr_oi"]
                    else:
                        exit_now = pcr_oi > trade["entry_pcr_oi"]

                # ---------- S2 (DELTA) ----------
                elif sid == "S2":
                    if opt_type == "CE":
                        exit_now = delta_bias <= trade["entry_delta"] - 5
                    else:
                        exit_now = delta_bias >= trade["entry_delta"] + 5

                # ---------- S3 (EMA CROSS) ----------
                elif sid == "S3":
                    exit_now = ema9 < ema21 if opt_type == "CE" else ema9 > ema21

                # ---------- S4 (TRUE COMBINED AND EXIT) ----------
                elif sid == "S4":
                    if opt_type == "CE":
                        exit_now = (
                            pcr_oi < trade["entry_pcr_oi"] and
                            delta_bias <= trade["entry_delta"] - 5 and
                            ema9 < ema21
                        )
                    else:
                        exit_now = (
                            pcr_oi > trade["entry_pcr_oi"] and
                            delta_bias >= trade["entry_delta"] + 5 and
                            ema9 > ema21
                        )
                # ---------- S5 (OI SENTIMENT CHANGE) ----------
                elif sid == "S5":
                    exit_price = get_exit_price(opt_type, trade["token"], trade["strike"])
                
                    sl_price = trade["entry_price"] * 0.70   # 30% SL
                    tgt_price = trade["entry_price"] * 1.30  # 30% TARGET

                    if opt_type == "CE":
                        # Exit CE only when OI is BEARISH AND (SL or TARGET)
                        exit_now = (
                            oi_sentiment == "BEARISH" and
                            (exit_price <= sl_price or exit_price >= tgt_price)
                        )
                    else:
                        # Exit PE only when OI is BULLISH AND (SL or TARGET)
                        exit_now = (
                            oi_sentiment == "BULLISH" and
                            (exit_price <= sl_price or exit_price >= tgt_price)
                        )                


                if exit_now or self.api_client.is_force_exit_time():
                    
                    # ==================================================
                    # 🟥 PLACE LIVE SELL ORDER (THIS IS YOUR REQUEST)sss
                    # ==================================================
                    if sid == "S5" and symbol.upper() in ALLOWED_S5_SYMBOLS:
                        self.api_client.place_order_api(
                            scripCode=trade["token"],
                            direction="SELL",
                            quantity=trade["qty"],
                            price=0,
                            strike_price=trade["strike"],
                            option_type=opt_type,
                            exchange=current_data.get("Exch", "N")
                        )


                    if exit_price <= 0:
                        print(
                            f"[WARN] Exit price not found | "
                            f"Symbol: {symbol} | Strategy: {sid} | Token: {trade['token']}"
                        )
                        continue


                    pnl = (exit_price - trade["entry_price"]) * trade["qty"]
                    self.db_manager.close_trade(trade["trade_id"], exit_price, pnl)
                    trade.update(new_trade_state())


    
    # def display_enhanced_analysis_latest(
    # self,
    # symbol: str,
    # current_price: float,
    # nearest_call: Dict[str, float],
    # nearest_put: Dict[str, float],
    # current_data: Dict,
    # option_analysis: Dict,
    # oc_supports: list,
    # oc_resistances: list,
    # market_bias: str,
    # validated_supports: list,
    # validated_resistances: list,
    # hist_df: pd.DataFrame
    # ):
    #     def get_exit_price(option_type, token, strike):
    #         options = option_analysis.get("option_chain", [])
    #         for opt in options:
    #             if opt.get("option_type") == option_type and (
    #                 opt.get("token") == token or opt.get("strike_price") == strike
    #             ):
    #                 return opt.get("last_price", 0)
    #         return 0


    #     last = hist_df.iloc[-1]

    #     tech = {
    #         "ema_9": round(last.get("EMA_9", 0), 2),
    #         "ema_21": round(last.get("EMA_21", 0), 2),
    #         "vwap": round(last.get("VWAP", 0), 2),
    #         "rsi": round(last.get("RSI", 0), 2),
    #         "macd_hist": round(last.get("MACD", 0), 2),
    #         "atr": round(last.get("ATR", 0), 2),
    #     }

    #     print(f"\n{'='*80}")
    #     print(f"MARKET ANALYSIS: {symbol}")
    #     print(f"{'='*80}")

    #     print(f"Price: Rs.{current_price:.2f}")
    #     print(
    #         f"O: {current_data.get('open', 0):.2f} | "
    #         f"H: {current_data.get('high', 0):.2f} | "
    #         f"L: {current_data.get('low', 0):.2f} | "
    #         f"Vol: {current_data.get('volume', 0):,}"
    #     )

    #     print(
    #         f"Nearest CALL: {nearest_call['strike_price']} @ {nearest_call['last_price']}"
    #         if nearest_call else "Nearest CALL: N/A"
    #     )
    #     print(
    #         f"Nearest PUT : {nearest_put['strike_price']} @ {nearest_put['last_price']}"
    #         if nearest_put else "Nearest PUT : N/A"
    #     )

    #     print(
    #         f"EMA9: {tech.get('ema_9', 'NA')} | "
    #         f"EMA21: {tech.get('ema_21', 'NA')}"
    #     )

    #     candle_high = current_data.get("high", 0)
    #     candle_low = current_data.get("low", 0)
    #     candle_close = current_price

    #     oi_sentiment = option_analysis.get("oi_sentiment")
    #     delta_bias = option_analysis.get("delta_bias_score", 50)
    #     oi_exit_price_new = None

    #     # ==================================================
    #     # INIT OI STATE
    #     # ==================================================
    #     if symbol not in CONFIG.oi_state:
    #         CONFIG.oi_state[symbol] = {
    #             "CE": {
    #                 "active": False,
    #                 "strike": None,
    #                 "token": None,
    #                 "entry_price": None,
    #                 "last_used_level": None
    #             },
    #             "PE": {
    #                 "active": False,
    #                 "strike": None,
    #                 "token": None,
    #                 "entry_price": None,
    #                 "last_used_level": None
    #             },
    #             "bullish_price_confirmed": False,
    #             "bullish_crossed_level": None,
    #             "bearish_price_confirmed": False,
    #             "bearish_crossed_level": None
    #         }

    #     state = CONFIG.oi_state[symbol]

    #     # ==================================================
    #     # 🟢 BULLISH PRICE CONFIRMATION → CE
    #     # ==================================================
    #     state["bullish_price_confirmed"] = False
    #     state["bullish_crossed_level"] = None

    #     hist_support_levels = []
    #     for sup in validated_supports:
    #         hist_support_levels.extend(sup.get("historical_matches", []))
    #     hist_support_levels = list(set(hist_support_levels))

    #     if hist_support_levels:
    #         crossed = [lvl for lvl in hist_support_levels if candle_low < lvl]
    #         if crossed:
    #             level = max(crossed)
    #             if candle_close > level:
    #                 state["bullish_price_confirmed"] = True
    #                 state["bullish_crossed_level"] = level

    #     # ==================================================
    #     # 🔴 BEARISH PRICE CONFIRMATION → PE
    #     # ==================================================
    #     state["bearish_price_confirmed"] = False
    #     state["bearish_crossed_level"] = None

    #     hist_resistance_levels = []
    #     for res in validated_resistances:
    #         hist_resistance_levels.extend(res.get("historical_matches", []))
    #     hist_resistance_levels = list(set(hist_resistance_levels))

    #     if hist_resistance_levels:
    #         crossed = [lvl for lvl in hist_resistance_levels if candle_high > lvl]
    #         if crossed:
    #             level = min(crossed)
    #             if candle_close < level:
    #                 state["bearish_price_confirmed"] = True
    #                 state["bearish_crossed_level"] = level

    #     ce_state = state["CE"]
    #     pe_state = state["PE"]

    #     # ==================================================
    #     # 🟢 CE ENTRY (BULLISH ONLY)
    #     # ==================================================
    #     if (
    #         not ce_state["active"] and
    #         oi_sentiment == "BULLISH" and
    #         state["bullish_price_confirmed"] and
    #         nearest_call and
    #         state["bullish_crossed_level"] != ce_state["last_used_level"]
    #     ):
    #         ce_state["active"] = True
    #         ce_state["strike"] = nearest_call["strike_price"]
    #         ce_state["token"] = nearest_call.get("token") or nearest_call["scripCode"]
    #         ce_state["entry_price"] = nearest_call["last_price"]
    #         ce_state["last_used_level"] = state["bullish_crossed_level"]

    #         print(f"🟢 BUY CE | {symbol} | {ce_state['strike']} @ {ce_state['entry_price']}")

    #         qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots
    #         self.api_client.place_order_api(
    #             scripCode=nearest_call["scripCode"],
    #             direction="BUY",
    #             quantity=qty,
    #             price=ce_state["entry_price"],
    #             strike_price=ce_state["strike"],
    #             option_type="CE"
    #         )

    #     # ==================================================
    #     # 🔴 PE ENTRY (BEARISH ONLY)
    #     # ==================================================
    #     if (
    #         not pe_state["active"] and
    #         oi_sentiment == "BEARISH" and
    #         state["bearish_price_confirmed"] and
    #         nearest_put and
    #         state["bearish_crossed_level"] != pe_state["last_used_level"]
    #     ):
    #         pe_state["active"] = True
    #         pe_state["strike"] = nearest_put["strike_price"]
    #         pe_state["token"] = nearest_put.get("token") or nearest_put["scripCode"]
    #         pe_state["entry_price"] = nearest_put["last_price"]
    #         pe_state["last_used_level"] = state["bearish_crossed_level"]

    #         print(f"🔴 BUY PE | {symbol} | {pe_state['strike']} @ {pe_state['entry_price']}")

    #         qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots
    #         self.api_client.place_order_api(
    #             scripCode=nearest_put["scripCode"],
    #             direction="BUY",
    #             quantity=qty,
    #             price=pe_state["entry_price"],
    #             strike_price=pe_state["strike"],
    #             option_type="PE"
    #         )

    #     # ==================================================
    #     # ❌ EXIT LOGIC
    #     # ==================================================
    #     if ce_state["active"] and oi_sentiment == "BEARISH":
    #         oi_exit_price_new = get_exit_price("CE", ce_state["token"], ce_state["strike"])
    #         print(f"❌ SELL CE | {symbol} | {ce_state['strike']} @ {oi_exit_price_new}")
    #         ce_state.update({"active": False, "strike": None, "token": None, "entry_price": None})


    #     if pe_state["active"] and oi_sentiment == "BULLISH":
    #         oi_exit_price_new = get_exit_price("PE", pe_state["token"], pe_state["strike"])
    #         print(f"❌ SELL PE | {symbol} | {pe_state['strike']} @ {oi_exit_price_new}")
    #         pe_state.update({"active": False, "strike": None, "token": None, "entry_price": None})


    #     # ==================================================
    #     # DB STORE
    #     # ==================================================
    #     active_trade = "CE" if ce_state["active"] else "PE" if pe_state["active"] else None
    #     active_state = ce_state if ce_state["active"] else pe_state if pe_state["active"] else {}

    #     self.db_manager.store_sentiment_trade(
    #         symbol=symbol,
    #         sentiment_type="OI",
    #         direction=active_trade,
    #         strike=active_state.get("strike"),
    #         token=active_state.get("token"),
    #         entry_price=active_state.get("entry_price"),
    #         exit_price=oi_exit_price_new,
    #         current_index_price=current_price,
    #         current_data=current_data,
    #         option_analysis=option_analysis
    #     )






    # def display_enhanced_analysis_latest(
    #     self,
    #     symbol: str,
    #     current_price: float,
    #     nearest_call: Dict[str, float],
    #     nearest_put: Dict[str, float],
    #     current_data: Dict,
    #     option_analysis: Dict,
    #     oc_supports: list,
    #     oc_resistances: list,
    #     market_bias: str,
    #     validated_supports: list,
    #     validated_resistances: list
    # ):
    #     print(f"\n{'='*80}")
    #     print(f"MARKET ANALYSIS: {symbol}")
    #     print(f"{'='*80}")

    #     print(f"Price: Rs.{current_price:.2f}")        
    #     print(f"O: {current_data.get('open', 0):.2f} | H: {current_data.get('high', 0):.2f} | L: {current_data.get('low', 0):.2f} | Vol: {current_data.get('volume', 0):,}")

    #     print(f"Nearest CALL strike: {nearest_call['strike_price'] if nearest_call else 'N/A'} | Last: {nearest_call['last_price'] if nearest_call else 'N/A'}")
    #     print(f"Nearest PUT strike: {nearest_put['strike_price'] if nearest_put else 'N/A'} | Last: {nearest_put['last_price'] if nearest_put else 'N/A'}")


    #     # ------------------------------------------------------------------
    #     # HIGH ACCURACY SUPPORT & RESISTANCE
    #     # ------------------------------------------------------------------
    #     # print(f"\n🎯 VALIDATED HIGH ACCURACY LEVELS (OI + Historical Validation):")
    #     # if validated_supports:
    #     #     print("  SUPPORTS:")
    #     #     for sup in validated_supports:
    #     #         info = f"Level: {sup['level']:.0f}"
    #     #         hist = f"Hist: {','.join([str(round(x,2)) for x in sup['historical_matches']])}" if sup['historical_matches'] else "Hist: -"
    #     #         swing = f"Swings: {','.join([str(round(x,2)) for x in sup['swing_matches']])}" if sup['swing_matches'] else "Swings: -"
    #     #         print(f"    {info} | {hist} | {swing}")
    #     # else:
    #     #     print("    No supports validated.")

    #     # if validated_resistances:
    #     #     print("  RESISTANCES:")
    #     #     for res in validated_resistances:
    #     #         info = f"Level: {res['level']:.0f}"
    #     #         hist = f"Hist: {','.join([str(round(x,2)) for x in res['historical_matches']])}" if res['historical_matches'] else "Hist: -"
    #     #         swing = f"Swings: {','.join([str(round(x,2)) for x in res['swing_matches']])}" if res['swing_matches'] else "Swings: -"
    #     #         print(f"    {info} | {hist} | {swing}")
    #     # else:
    #     #     print("    No resistances validated.")


    #     # --------------------------------------------------
    #     # 🎯 TRADE CONFIRMATION (HIST LEVEL CROSS + REJECTION)
    #     # --------------------------------------------------

    #     candle_high  = current_data.get("high", 0)
    #     candle_low   = current_data.get("low", 0)
    #     candle_close = current_price

    #     oi_sentiment = option_analysis.get("oi_sentiment")
    #     delta_bias   = option_analysis.get("delta_bias_score", 50)
    #     oi_exit_price_new = None
    #     #================================= Test =================



    #     # --------------------------------------------------
    #     # OI STATE INIT
    #     # --------------------------------------------------
    #     if symbol not in CONFIG.oi_state:
    #         CONFIG.oi_state[symbol] = {
    #             "oi_trade": None,
    #             "oi_strike": None,
    #             "oi_token": None,
    #             "oi_entry_price": None,
    #             # 🔹 PRICE CONFIRMATION STATE
    #             "bullish_price_confirmed": False,
    #             "bullish_crossed_level": None,
    #             "bearish_price_confirmed": False,
    #             "bearish_crossed_level": None,
    #             "last_used_bullish_level": None,
    #             "last_used_bearish_level": None
    #         }

    #     state = CONFIG.oi_state[symbol]
        
        
    #     # --------------------------------------------------
    #     # 🟢 BULLISH PRICE CONFIRMATION (SUPPORT RECLAIM)
    #     # --------------------------------------------------

    #     hist_support_levels = []
    #     for sup in validated_supports:
    #         hist_support_levels.extend(sup.get("historical_matches", []))

    #     hist_support_levels = list(set(hist_support_levels))

    #     if hist_support_levels:
    #         crossed_levels = [h for h in hist_support_levels if candle_low < h]
    #         if crossed_levels:
    #             state["bullish_crossed_level"] = max(crossed_levels)
    #             if candle_close > state["bullish_crossed_level"]:
    #                 state["bullish_price_confirmed"] = True

    #     # --------------------------------------------------
    #     # 🔴 BEARISH PRICE CONFIRMATION (RESISTANCE REJECTION)
    #     # --------------------------------------------------

    #     hist_resistance_levels = []
    #     for res in validated_resistances:
    #         hist_resistance_levels.extend(res.get("historical_matches", []))

    #     hist_resistance_levels = list(set(hist_resistance_levels))

    #     if hist_resistance_levels:
    #         crossed_levels = [h for h in hist_resistance_levels if candle_high > h]
    #         if crossed_levels:
    #             state["bearish_crossed_level"] = min(crossed_levels)
    #             if candle_close < state["bearish_crossed_level"]:
    #                 state["bearish_price_confirmed"] = True

    #     if state["oi_trade"] is None:
    #             # ENTRY
    #             print(f"ENTRY SIGNAL | {symbol} | OI Sentiment: {oi_sentiment} | Delta Bias: {delta_bias}")
    #             if oi_sentiment == "BULLISH" and state["bullish_price_confirmed"] and nearest_call and state["bullish_crossed_level"] != state["last_used_bullish_level"]:
    #                 state["bearish_price_confirmed"] = False
    #                 state["bearish_crossed_level"] = None
    #                 state["last_used_bullish_level"] = state["bullish_crossed_level"]
    #                 state["oi_trade"] = "CE"
    #                 state["oi_strike"] = nearest_call.get("strike_price")
    #                 state["oi_token"] = nearest_call.get("token") or nearest_call.get("scripCode")
    #                 state["oi_entry_price"] = nearest_call.get("last_price", 0.0)
    #                 print(f"BUY CE SIGNAL | {symbol} | Strike {nearest_call['strike_price']} | Price {nearest_call['last_price']}")

    #                 ce_scrip_code = nearest_call["scripCode"]
    #                 ce_price = nearest_call["last_price"]

    #                 qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots

    #                 self.api_client.place_order_api(
    #                         scripCode=ce_scrip_code,
    #                         direction="BUY",
    #                         quantity=qty,
    #                         price=ce_price,
    #                         strike_price=nearest_call["strike_price"],
    #                         option_type="CE"
    #                 )
    #             elif oi_sentiment == "BEARISH" and state["bearish_price_confirmed"] and nearest_put and state["bearish_crossed_level"] != state["last_used_bearish_level"]:
    #                 state["bullish_price_confirmed"] = False
    #                 state["bullish_crossed_level"] = None
    #                 state["last_used_bearish_level"] = state["bearish_crossed_level"]
    #                 state["oi_trade"] = "PE"
    #                 state["oi_strike"] = nearest_put.get("strike_price")
    #                 state["oi_token"] = nearest_put.get("token") or nearest_put.get("scripCode")
    #                 state["oi_entry_price"] = nearest_put.get("last_price", 0.0)
    #                 print(f"BUY PE SIGNAL | {symbol} | Strike {nearest_put['strike_price']} | Price {nearest_put['last_price']}")
                        
    #                 pe_scrip_code = nearest_put["scripCode"]
    #                 pe_price = nearest_put["last_price"]

    #                 qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots

    #                 self.api_client.place_order_api(
    #                         scripCode=pe_scrip_code,
    #                         direction="BUY",
    #                         quantity=qty,
    #                         price=pe_price,
    #                         strike_price=nearest_put["strike_price"],
    #                         option_type="PE"
    #                     )
    #     else:
    #             # EXIT on sentiment flip
    #             if state["oi_trade"] == "CE" and oi_sentiment == "BEARISH":
    #                 print(f"SELL CE SIGNAL | {symbol} | Strike {state['oi_strike']} | Price {nearest_call.get("last_price", 0.0)}")
    #                 oi_exit_price_new = nearest_call.get("last_price", 0.0)
    #                 state["oi_trade"] = None
    #                 state["oi_strike"] = None
    #                 state["oi_token"] = None
    #                 state["oi_entry_price"] = None

    #             elif state["oi_trade"] == "PE" and oi_sentiment == "BULLISH":
    #                 print(f"SELL PE SIGNAL | {symbol} | Strike {state['oi_strike']} | Price {nearest_put.get('last_price', 0.0)}")
    #                 oi_exit_price_new = nearest_put.get("last_price", 0.0)
    #                 state["oi_trade"] = None
    #                 state["oi_strike"] = None
    #                 state["oi_token"] = None
    #                 state["oi_entry_price"] = None

    #     self.db_manager.store_sentiment_trade(
    #         symbol=symbol,
    #         sentiment_type="OI",
    #         direction=state["oi_trade"],
    #         strike=state["oi_strike"],
    #         token=state["oi_token"],
    #         entry_price=state["oi_entry_price"],
    #         exit_price=oi_exit_price_new,
    #         current_index_price=current_price,
    #         current_data=current_data,
    #         option_analysis=option_analysis
    #     )

    #     # STORE TO DB
    #     self.db_manager.store_market_analysis_summary(
    #         symbol=symbol,
    #         current_price=current_price,
    #         current_data={
    #             'open': current_data.get('open'),
    #             'high': current_data.get('high'),
    #             'low': current_data.get('low'),
    #             'volume': current_data.get('volume')
    #         },
    #         market_bias=f"{option_analysis.get('sentiment', 'NEUTRAL')} "
    #                     f"pcr oi : {option_analysis.get('pcr_oi', 0)} "
    #                     f"pcr oi change : {option_analysis.get('pcr_oi_change', 0)} "
    #                     f"pcr volume : {option_analysis.get('pcr_volume', 0)} "
    #                     f"pcr_vol_price : {option_analysis.get('pcr_vol_price', 0)} ",
    #         nearest_call={'strike_price': nearest_call['strike_price'], 'last_price': nearest_call['last_price']},
    #         nearest_put={'strike_price': nearest_put['strike_price'], 'last_price': nearest_put['last_price']}
    #     )
        
        

        # OPTION CHAIN SUMMARY
        # if option_analysis:
        #     print(f"\nOPTION CHAIN:")
        #     print(f"  Calls - Vol: {option_analysis.get('call_volume', 0):,} | OI: {option_analysis.get('call_oi', 0):,} | OI Δ: {option_analysis.get('call_oi_change', 0):,}")
        #     print(f"  Puts  - Vol: {option_analysis.get('put_volume', 0):,} | OI: {option_analysis.get('put_oi', 0):,} | OI Δ: {option_analysis.get('put_oi_change', 0):,}")
        #     print(f"  PCR OI: {option_analysis.get('pcr_oi', 0):.5f} | PCR OI Chg: {option_analysis.get('pcr_oi_change', 0):.5f} | "
        #         f"PCR Vol: {option_analysis.get('pcr_volume', 0):.5f} | PCR Vol Price: {option_analysis.get('pcr_vol_price', 0):.5f}")
        #     print(f"  Combined SENTIMENT: {option_analysis.get('sentiment', 'NEUTRAL')}\n")

        # ------------------------------------------------------------------
        # 🔥 DISPLAY SENTIMENT-BASED TRADES WITH ENTRY/EXIT PRICE
        # ------------------------------------------------------------------
        # print("📌 SENTIMENT TRADE STATUS")

        # # =============== OI TRADE ==================
        # print("\n👉 OI Sentiment Trade:")
        # print(f"   Sentiment: {option_analysis.get('oi_sentiment')}")
        # print(f"   Trade: {option_analysis.get('oi_trade')}")
        # print(f"   Strike: {option_analysis.get('oi_strike')}")
        # print(f"   Token: {option_analysis.get('oi_token')}")
        # print(f"   Entry Price: {option_analysis.get('oi_entry_price')}")
        # print(f"   Exit Price: {option_analysis.get('oi_exit_price')}")

        # # =============== OI CHANGE TRADE ==================
        # print("\n👉 OI Change Sentiment Trade:")
        # print(f"   Sentiment: {option_analysis.get('oi_change_sentiment')}")
        # print(f"   Trade: {option_analysis.get('oi_change_trade')}")
        # print(f"   Strike: {option_analysis.get('oi_change_strike')}")
        # print(f"   Token: {option_analysis.get('oi_change_token')}")
        # print(f"   Entry Price: {option_analysis.get('oi_change_entry_price')}")
        # print(f"   Exit Price: {option_analysis.get('oi_change_exit_price')}")

        # # =============== VOLUME TRADE ==================
        # print("\n👉 Volume Sentiment Trade:")
        # print(f"   Sentiment: {option_analysis.get('volume_sentiment')}")
        # print(f"   Trade: {option_analysis.get('volume_trade')}")
        # print(f"   Strike: {option_analysis.get('volume_strike')}")
        # print(f"   Token: {option_analysis.get('volume_token')}")
        # print(f"   Entry Price: {option_analysis.get('volume_entry_price')}")
        # print(f"   Exit Price: {option_analysis.get('volume_exit_price')}")



        # ------------------------------------------------------------------
        # OI ONLY S/R
        # ------------------------------------------------------------------
        # print(f"\nKEY OI LEVELS:")
        # if oc_supports:
        #     print(f"  OI Supports: {' | '.join([str(int(s)) for s in oc_supports[:3]])}")
        # if oc_resistances:
        #     print(f"  OI Resistances: {' | '.join([str(int(r)) for r in oc_resistances[:3]])}")

        # print(f"{'='*80}\n")


    
    # def display_enhanced_analysis_latest(
    #     self,
    #     symbol: str,
    #     current_price: float,
    #     nearest_call: Dict[str, float],
    #     nearest_put: Dict[str, float],
    #     current_data: Dict,
    #     option_analysis: Dict,
    #     oc_supports: list,
    #     oc_resistances: list,
    #     market_bias: str,
    #     validated_supports: list,
    #     validated_resistances: list
    # ):
    #     print(f"\n{'='*80}")
    #     print(f"MARKET ANALYSIS: {symbol}")
    #     print(f"{'='*80}")
    #     print(f"Price: Rs.{current_price:.2f}")        
    #     print(f"O: {current_data.get('open', 0):.2f} | H: {current_data.get('high', 0):.2f} | L: {current_data.get('low', 0):.2f} | Vol: {current_data.get('volume', 0):,}")
    #     print(f"Market Bias (OC OI): {market_bias}")    
    #     print(f"Nearest CALL strike: {nearest_call['strike_price'] if nearest_call else 'N/A'} | Last: {nearest_call['last_price'] if nearest_call else 'N/A'}")
    #     print(f"Nearest PUT strike: {nearest_put['strike_price'] if nearest_put else 'N/A'} | Last: {nearest_put['last_price'] if nearest_put else 'N/A'}")


    #     self.db_manager.store_market_analysis_summary(
    #     symbol=symbol,
    #     current_price=current_price,
    #     current_data={'open': current_data.get('open'), 'high': current_data.get('high'), 'low': current_data.get('low'), 'volume': current_data.get('volume')},
    #     market_bias=f"{option_analysis.get('sentiment', 'NEUTRAL')} pcr oi : {option_analysis.get('pcr_oi', 0)} pcr oi change : {option_analysis.get('pcr_oi_change', 0)} pcr volume : {option_analysis.get('pcr_volume', 0)} pcr_vol_price : {option_analysis.get('pcr_vol_price', 0)} buy_action : {option_analysis.get('buy_action', 0)} buy_strike : {option_analysis.get('buy_strike', 0)} buy_strike_code : {option_analysis.get('buy_strike_code', 0)} ",
    #     nearest_call={'strike_price': nearest_call['strike_price'], 'last_price': nearest_call['last_price']},
    #     nearest_put={'strike_price': nearest_put['strike_price'], 'last_price': nearest_put['last_price']}
    #     )

    #     # Option Chain Analysis
    #     if option_analysis:
    #         print(f"\nOPTION CHAIN:")
    #         print(f"  Calls - Vol: {option_analysis.get('call_volume', 0):,} | OI: {option_analysis.get('call_oi', 0):,} | OI Δ: {option_analysis.get('call_oi_change', 0):,}")
    #         print(f"  Puts  - Vol: {option_analysis.get('put_volume', 0):,} | OI: {option_analysis.get('put_oi', 0):,} | OI Δ: {option_analysis.get('put_oi_change', 0):,}")
    #         print(f"  PCR OI: {option_analysis.get('pcr_oi', 0):.5f} PCR OI Change: {option_analysis.get('pcr_oi_change', 0):.5f} PCR volume: {option_analysis.get('pcr_volume', 0):.5f} | PCR Vol Price: {option_analysis.get('pcr_vol_price', 0):.5f}| SENTIMENT: {option_analysis.get('sentiment', 'NEUTRAL')}")

    #     # High Accuracy Support and Resistance Section
    #     print(f"\n🎯 VALIDATED HIGH ACCURACY LEVELS (OI + Historical Validation):")
    #     if validated_supports:
    #         print("  SUPPORTS:")
    #         for sup in validated_supports:
    #             info = f"Level: {sup['level']:.0f}"
    #             hist = f"Hist: {','.join([str(round(x,2)) for x in sup['historical_matches']])}" if sup['historical_matches'] else "Hist: -"
    #             swing = f"Swings: {','.join([str(round(x,2)) for x in sup['swing_matches']])}" if sup['swing_matches'] else "Swings: -"
    #             print(f"    {info} | {hist} | {swing}")
    #     else:
    #         print("    No supports validated.")

    #     if validated_resistances:
    #         print("  RESISTANCES:")
    #         for res in validated_resistances:
    #             info = f"Level: {res['level']:.0f}"
    #             hist = f"Hist: {','.join([str(round(x,2)) for x in res['historical_matches']])}" if res['historical_matches'] else "Hist: -"
    #             swing = f"Swings: {','.join([str(round(x,2)) for x in res['swing_matches']])}" if res['swing_matches'] else "Swings: -"
    #             print(f"    {info} | {hist} | {swing}")
    #     else:
    #         print("    No resistances validated.")

    #     # OI Only S/R
    #     print(f"\nKEY OI LEVELS:")
    #     if oc_supports:
    #         print(f"  OI Supports: {' | '.join([str(int(s)) for s in oc_supports[:3]])}")
    #     if oc_resistances:
    #         print(f"  OI Resistances: {' | '.join([str(int(r)) for r in oc_resistances[:3]])}")

    #     print(f"{'='*80}\n")

    def display_enhanced_analysis(self, symbol: str, current_price: float, current_data: Dict,
                                option_analysis: Dict, oc_supports: List, oc_resistances: List,
                                high_accuracy_supports: List, high_accuracy_resistances: List) -> None:
        """Display comprehensive analysis with high accuracy levels."""
        
        print(f"\n{'-'*80}")
        print(f"MARKET ANALYSIS: {symbol}")
        print(f"{'-'*80}")
        print(f"Price: Rs.{current_price:.2f} | O: {current_data.get('close', 0):.2f} | "
              f"H: {current_data.get('high', 0):.2f} | L: {current_data.get('low', 0):.2f} | "
              f"Vol: {current_data.get('volume', 0):,}")
        
        # Option Chain Analysis
        if option_analysis:
            print(f"\nOPTION CHAIN:")
            print(f"  Calls - Vol: {option_analysis.get('call_volume', 0):,} | "
                  f"OI: {option_analysis.get('call_oi', 0):,} | "
                  f"OI Δ: {option_analysis.get('call_oi_change', 0):,}")
            print(f"  Puts  - Vol: {option_analysis.get('put_volume', 0):,} | "
                  f"OI: {option_analysis.get('put_oi', 0):,} | "
                  f"OI Δ: {option_analysis.get('put_oi_change', 0):,}")
            print(f"  PCR: {option_analysis.get('pcr_oi', 0):.3f} | "
                  f"SENTIMENT: {option_analysis.get('sentiment', 'NEUTRAL')}")
        
        # NEW: High Accuracy Support and Resistance Section
        if high_accuracy_supports or high_accuracy_resistances:
            print(f"\n🎯 HIGH ACCURACY LEVELS (OI + Historical Validation):")
            
            if high_accuracy_supports:
                print(f"  SUPPORTS:")
                for sup in high_accuracy_supports:
                    print(f"    {sup['level']:.0f} - Score: {sup['accuracy_score']} | "
                          f"Touches: {sup['touch_count']} | Swings: {sup['swing_confirmations']} | "
                          f"Strength: {sup['strength']}")
            
            if high_accuracy_resistances:
                print(f"  RESISTANCES:")
                for res in high_accuracy_resistances:
                    print(f"    {res['level']:.0f} - Score: {res['accuracy_score']} | "
                          f"Touches: {res['touch_count']} | Swings: {res['swing_confirmations']} | "
                          f"Strength: {res['strength']}")
        
        # Standard Support and Resistance
        print(f"\nKEY LEVELS:")
        if oc_supports:
            print(f"  OI Supports: {' | '.join([f'{s:.0f}' for s in oc_supports[:3]])}")
        if oc_resistances:
            print(f"  OI Resistance: {' | '.join([f'{r:.0f}' for r in oc_resistances[:3]])}")
        # if hist_supports:
        #     print(f"  Price Supports: {' | '.join([f'{s:.0f}' for s in hist_supports[:3]])}")
        # if hist_resistances:
        #     print(f"  Price Resistance: {' | '.join([f'{r:.0f}' for r in hist_resistances[:3]])}")
            
        # if hist_supports15:
        #     print(f"  Price Supports 15: {' | '.join([f'{s:.0f}' for s in hist_supports15[:10]])}")
        # if hist_resistances15:
        #     print(f"  Price Resistance 15: {' | '.join([f'{r:.0f}' for r in hist_resistances15[:10]])}")
        
        print(f"{'-'*80}")

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    """Main execution function."""
    try:
        print("\n" + "="*80)
        print("PROFESSIONAL TRADING SYSTEM v2.2")
        print("="*80)



        # Initialize and run trading system
        trading_system = SimplifiedTradingSystem()
        trading_system.run_analysis_loop()

        
    except Exception as e:
        Logger.error(f"System startup error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
