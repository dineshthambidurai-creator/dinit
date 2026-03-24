"""
Professional Trading System v5.0
Strategies:
  S1 - Zone Bounce
  S2 - EMA 9/21 Crossover + OI Sentiment
  S3 - VWAP Reclaim/Break + RSI Momentum
  S4 - Bollinger Band Breakout + Volume Surge + OI Sentiment
  S5 - RSI Extreme at Validated Support/Resistance
  S6 - MACD Zero-Line Cross + Stochastic Alignment
  S7 - 9:30 Opening Range Breakout (ORB)
  S8 - Swing High/Low + Trendline Break + Candlestick Pattern

TARGET LOGIC (per strategy, added in v5):
  Every strategy now has:
    • sl_price   — stop loss stored in trade state and DB
    • target_price — profit target stored in trade state and DB
    • Exit fires on: signal reversal OR SL hit OR TARGET hit OR force EOD
  Risk:Reward ratios per strategy:
    S1: 1:1.5  (zone bounce - tight)
    S2: 1:2.0  (EMA cross - trend following)
    S3: 1:1.5  (VWAP - mean reversion)
    S4: 1:2.5  (BB breakout - momentum)
    S5: 1:2.0  (RSI extreme - reversal)
    S6: 1:2.0  (MACD - momentum)
    S7: 1:2.0  (ORB - breakout)
    S8: 1:2.0  (Swing/Trendline - structure)
"""
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

import json
import os
import sys
import time
import warnings
import math
import re
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque
from statistics import mean

import numpy as np
import pandas as pd
import pyotp
import ta
from py5paisa import FivePaisaClient
import trendln
from scipy.stats import norm
from turso_db import get_db

  
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────
# RISK : REWARD RATIOS  (SL multiplier → Target multiplier)
# ───────────────────────────────────────────────────────────
RR = {
    "S1": 1.5,
    "S2": 2.0,
    "S3": 1.5,
    "S4": 2.5,
    "S5": 2.0,
    "S6": 2.0,
    "S7": 2.0,
    "S8": 2.0,
}

# ===============================
# UTILITY CLASSES
# ===============================

class SuppressPrints:
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
    @staticmethod
    def _log(level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp} | {level} | {message}", flush=True)

    @staticmethod
    def info(message):    Logger._log("INFO",    message)
    @staticmethod
    def success(message): Logger._log("SUCCESS", message)
    @staticmethod
    def error(message):   Logger._log("ERROR",   message)
    @staticmethod
    def warning(message): Logger._log("WARNING", message)

# ===============================
# CONFIGURATION
# ===============================

@dataclass
class TradingConfig:
    SYMBOLS: List[str] = None
    STRIKE_STEPS: Dict[str, int] = None
    LOT_SIZES: Dict[str, int] = None
    DATABASE_PATH: str = "trading_data.db"
    DATA_UPDATE_INTERVAL: int = 23
    number_of_lots: int = 1
    oi_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    orb_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    delta_oi_state: Dict[str, Any] = field(default_factory=lambda: {
        "active": False,
        "side": None,
        "entry_price": None
    })

    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX']

        if self.STRIKE_STEPS is None:
            self.STRIKE_STEPS = {
                'NIFTY': 50, 'BANKNIFTY': 100, 'BANKEX': 100,
                'FINNIFTY': 50, 'SENSEX': 100, 'MIDCPNIFTY': 50
            }

        if self.LOT_SIZES is None:
            self.LOT_SIZES = {
                'NIFTY': 65, 'BANKNIFTY': 30, 'FINNIFTY': 60,
                'MIDCPNIFTY': 120, 'SENSEX': 20
            }

CONFIG = TradingConfig()

# ===============================
# TRADE STATE HELPER
# ===============================

def new_trade_state():
    return {
        "active":          False,
        "trade_id":        None,
        "strike":          None,
        "token":           None,
        "symbol":          None,
        "entry_price":     None,
        "qty":             None,
        "last_used_level": None,
        "entry_oi":        None,
        "entry_delta":     None,
        # S7 ORB
        "orb_high":        None,
        "orb_low":         None,
        # S8 Swing
        "swing_level":     None,
        "pattern":         None,
        # TARGETS & SL (all strategies)
        "sl_price":        None,
        "target_price":    None,
    }

# ===============================
# DATABASE MANAGER
# ===============================

class DatabaseManager:
    def __init__(self):
        self.db = get_db()
        self._initialize_database()

    # -----------------------------
    # INIT TABLES
    # -----------------------------
    def _initialize_database(self):

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS option_chain_data (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            strike_price REAL,
            option_type TEXT,
            last_price REAL,
            bid REAL,
            ask REAL,
            volume INTEGER,
            open_interest INTEGER,
            change_in_oi INTEGER,
            implied_volatility REAL
        )
        """)

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            open_price REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
        """)

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS option_trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
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
            orb_high REAL,
            orb_low REAL,
            sl_price REAL,
            target_price REAL,
            swing_level REAL,
            pattern TEXT,
            exit_reason TEXT,
            entry_time TEXT,
            exit_time TEXT
        )
        """)

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS market_analysis_summary (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
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

    # -----------------------------
    # STORE OPTION CHAIN
    # -----------------------------
    def store_option_chain_data(self, symbol, option_chain_data):

        queries = []

        for opt in option_chain_data:
            queries.append((
                """INSERT INTO option_chain_data
                (timestamp, symbol, strike_price, option_type,
                 last_price, bid, ask, volume, open_interest,
                 change_in_oi, implied_volatility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    datetime.now().isoformat(),
                    symbol,
                    opt.get('strike_price', 0),
                    opt.get('option_type', ''),
                    opt.get('last_price', 0),
                    opt.get('bid', 0),
                    opt.get('ask', 0),
                    opt.get('volume', 0),
                    opt.get('open_interest', 0),
                    opt.get('change_in_oi', 0),
                    opt.get('implied_volatility', 0)
                ]
            ))

        if queries:
            self.db.batch(queries)

    # -----------------------------
    # STORE MARKET DATA
    # -----------------------------
    def store_market_data(self, symbol, market_data):

        self.db.execute("""
        INSERT INTO market_data
        (timestamp, symbol, open_price, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            datetime.now().isoformat(),
            symbol,
            market_data.get('open', 0),
            market_data.get('high', 0),
            market_data.get('low', 0),
            market_data.get('close', 0),
            market_data.get('volume', 0)
        ])

    # -----------------------------
    # INSERT TRADE
    # -----------------------------
    def insert_trade(self, **data):

        result = self.db.execute("""
        INSERT INTO option_trades (
            symbol, strategy, option_type, strike, token,
            qty, entry_price, status, entry_oi, entry_delta,
            orb_high, orb_low,
            sl_price, target_price,
            swing_level, pattern,
            entry_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            data["symbol"],
            data["strategy"],
            data["option_type"],
            data["strike"],
            data["token"],
            data["qty"],
            data["entry_price"],
            data["entry_oi"],
            data["entry_delta"],
            data.get("orb_high"),
            data.get("orb_low"),
            data.get("sl_price"),
            data.get("target_price"),
            data.get("swing_level"),
            data.get("pattern"),
            datetime.now().isoformat()
        ])

        return result.last_insert_rowid

    # -----------------------------
    # CLOSE TRADE
    # -----------------------------
    def close_trade(self, trade_id, exit_price, pnl, exit_reason="SIGNAL_EXIT"):

        self.db.execute("""
        UPDATE option_trades
        SET exit_price=?, pnl=?, status='CLOSED',
            exit_reason=?, exit_time=?
        WHERE id=?
        """, [
            exit_price,
            pnl,
            exit_reason,
            datetime.now().isoformat(),
            trade_id
        ])

    # -----------------------------
    # RESTORE OPEN TRADES
    # -----------------------------
    def restore_open_trades(self):

        result = self.db.execute(
            "SELECT * FROM option_trades WHERE status='OPEN'"
        )

        rows = result.rows

        Logger.info(f"Restored {len(rows)} open trades")

        return rows

    # -----------------------------
    # MARKET SUMMARY
    # -----------------------------
    def store_market_analysis_summary(self, symbol, current_price, current_data,
                                      market_bias, nearest_call, nearest_put):

        self.db.execute("""
        INSERT INTO market_analysis_summary (
            timestamp, symbol, current_price,
            market_open, market_high, market_low, market_volume, market_bias,
            nearest_call_strike, nearest_call_last,
            nearest_put_strike, nearest_put_last
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            datetime.now().isoformat(),
            symbol,
            current_price,
            current_data.get('open', 0),
            current_data.get('high', 0),
            current_data.get('low', 0),
            current_data.get('volume', 0),
            market_bias,
            nearest_call.get('strike_price', 0) if nearest_call else None,
            nearest_call.get('last_price', 0) if nearest_call else None,
            nearest_put.get('strike_price', 0) if nearest_put else None,
            nearest_put.get('last_price', 0) if nearest_put else None
        ])
# ===============================
# API CLIENT
# ===============================

class APIClient:
    def __init__(self, credentials_path=None):
        self.client: Optional[FivePaisaClient] = None
        self.credentials_path = credentials_path
        self._scrips_cache: Optional[pd.DataFrame] = None
        self._open_fix_state: Dict[str, bool] = {}
        self._daily_open_cache: Dict[str, Any] = {}

    def _load_credentials(self):
        if self.credentials_path and os.path.exists(self.credentials_path):
            with open(self.credentials_path, "r") as f:
                cred = json.load(f)
        else:
            cred = {
                "CLIENT_CODE":    os.getenv("CLIENT_CODE"),
                "PIN":            os.getenv("PIN"),
                "TOTP_SECRET":    os.getenv("TOTP_SECRET"),
                "APP_NAME":       os.getenv("APP_NAME"),
                "APP_SOURCE":     os.getenv("APP_SOURCE"),
                "USER_ID":        os.getenv("USER_ID"),
                "PASSWORD":       os.getenv("PASSWORD"),
                "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY"),
                "USER_KEY":       os.getenv("USER_KEY")
            }
        missing = [k for k, v in cred.items() if not v]
        if missing:
            raise RuntimeError(f"Missing credentials: {missing}")
        return cred

    def initialize_client(self):
        try:
            cred = self._load_credentials()
            client = FivePaisaClient(cred=cred)
            totp = pyotp.TOTP(cred["TOTP_SECRET"])
            remaining = 30 - (int(time.time()) % 30)
            if remaining < 5:
                time.sleep(remaining + 1)
            with SuppressPrints():
                session = client.get_totp_session(
                    client_code=cred["CLIENT_CODE"],
                    totp=totp.now(),
                    pin=cred["PIN"]
                )
            if session:
                self.client = client
                return True
            Logger.error("Failed to create session")
            return False
        except Exception as e:
            Logger.error(f"API init failed: {e}")
            return False

    def load_scrips_data(self, file_path="scrips_data.json"):
      if self._scrips_cache is not None:
          return self._scrips_cache
  
      try:
          if not self.client:
              Logger.error("Client not initialized")
              return None
  
          Logger.info("Fetching scrips from API...")
  
          with SuppressPrints():
              scrips_live = self.client.get_scrips()
              Logger.info("Fetching scrips from API...")
  
          if scrips_live is not None and not scrips_live.empty:
              self._scrips_cache = scrips_live
              Logger.success(f"Scrips loaded: {len(scrips_live)}")
              return scrips_live
  
          Logger.error("Empty scrips response")
  
      except Exception as e:
          Logger.error(f"Load scrips failed: {e}")
  
      return None

    def find_scrip_info(self, symbol):
        scrips_df = self.load_scrips_data()
        if scrips_df is None or scrips_df.empty:
            return None
        scrips_df['ScripCode'] = scrips_df['ScripCode'].astype(str)
        if isinstance(symbol, int) or str(symbol).isdigit():
            match = scrips_df[scrips_df['ScripCode'] == str(symbol)]
            if not match.empty:
                return match.iloc[0]
        symbol_upper = str(symbol).upper()
        for exch in ['N', 'B']:
            for exch_type in ['I', 'C', 'D']:
                match = scrips_df[
                    (scrips_df['Name'].str.upper() == symbol_upper) &
                    (scrips_df['Exch'] == exch) &
                    (scrips_df['ExchType'] == exch_type)
                ]
                if not match.empty:
                    return match.iloc[0]
        Logger.error(f"Scrip not found: {symbol}")
        return None

    def get_current_price(self, symbol):
        scrip_info = self.find_scrip_info(symbol)
        if scrip_info is None:
            return None
        try:
            req_list = [{"Exch": scrip_info['Exch'],
                         "ExchType": scrip_info['ExchType'],
                         "ScripCode": int(scrip_info['ScripCode'])}]
            with SuppressPrints():
                market_data = self.client.fetch_market_feed_scrip(req_list)
            if market_data and 'Data' in market_data and len(market_data['Data']) > 0:
                pd_ = market_data['Data'][0]
                ltp = float(pd_.get('LastRate', 0))
                now = datetime.now()
                after_914 = (now.hour > 9) or (now.hour == 9 and now.minute >= 14)
                today_key = f"{symbol}_{now.date()}"
                if after_914 and not self._open_fix_state.get(today_key) and ltp > 0:
                    open_price = ltp
                    self._open_fix_state[today_key] = True
                else:
                    open_price = float(pd_.get('LastRate', 0))
                return {
                    'open': open_price,
                    'high': float(pd_.get('High', 0)),
                    'low': float(pd_.get('Low', 0)),
                    'close': ltp,
                    'pclose': float(pd_.get('PClose', 0)),
                    'volume': int(pd_.get('Volume', 0)),
                    'ltp': ltp,
                    'Exch': pd_.get('Exch', 'N')
                }
        except Exception as e:
            Logger.error(f"get_current_price failed for {symbol}: {e}")
        return None

    def is_trading_time(self):
        now = datetime.now()
        return dtime(9, 28) <= now.time() <= dtime(15, 0)

    def is_live_entry_time(self):
        return dtime(9, 30) <= datetime.now().time() <= dtime(15, 0)

    def is_force_exit_time(self):
        return dtime(15, 5) <= datetime.now().time() <= dtime(15, 15)

    def fetch_historical_data(self, symbol, mins, days_back=100):
        scrip_info = self.find_scrip_info(symbol)
        if scrip_info is None:
            return None
        try:
            end_date   = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            with SuppressPrints():
                hist_data = self.client.historical_data(
                    Exch=scrip_info['Exch'],
                    ExchangeSegment=scrip_info['ExchType'],
                    ScripCode=int(scrip_info['ScripCode']),
                    time=mins,
                    From=start_date,
                    To=end_date
                )
            if hist_data is not None and not hist_data.empty:
                return hist_data
        except Exception as e:
            Logger.error(f"fetch_historical_data failed for {symbol}: {e}")
        return None

    def get_option_chain_data(self, symbol):
        if not self.client:
            return []
        try:
            scrip_info  = self.find_scrip_info(symbol)
          
            if scrip_info is None:
              Logger.error(f"Scrip info not found for {symbol}")
              return []
            
            expiry_data = self.client.get_expiry(scrip_info['Exch'], symbol)
          
            if not expiry_data:
              Logger.error(f"No expiry data for {symbol}")
              return []

            if 'Expiry' not in expiry_data or not expiry_data['Expiry']:
                Logger.error(f"Invalid expiry data for {symbol}")
                return []
              
            expiry_date_str = expiry_data['Expiry'][0]['ExpiryDate']
            match = re.search(r'/Date\((\d+)', expiry_date_str)
            if not match:
                return []
            expiry_timestamp = int(match.group(1))
            option_chain = self.client.get_option_chain(scrip_info['Exch'], symbol, expiry_timestamp)
            if isinstance(option_chain, dict):
                for key in ['Options', 'Data', 'OptionChain']:
                    if key in option_chain:
                        return self._process_option_chain_data(option_chain[key], expiry_timestamp)
            elif hasattr(option_chain, 'to_dict'):
                return self._process_option_chain_data(option_chain.to_dict('records'), expiry_timestamp)
        except Exception as e:
            Logger.error(f"get_option_chain_data failed for {symbol}: {e}")
        return []

    def _process_option_chain_data(self, raw_data, expiry_timestamp):
        processed = []
        for opt in raw_data:
            try:
                processed.append({
                    'name':              opt.get('Name', ''),
                    'ScripCode':         int(opt.get('ScripCode', 0)),
                    'strike_price':      float(opt.get('StrikeRate', 0)),
                    'option_type':       opt.get('CPType', ''),
                    'last_price':        float(opt.get('LastRate', 0)),
                    'bid':               float(opt.get('BidPrice', 0)),
                    'ask':               float(opt.get('AskPrice', 0)),
                    'volume':            int(opt.get('Volume', 0)),
                    'open_interest':     int(opt.get('OpenInterest', 0)),
                    'change_in_oi':      int(opt.get('ChangeInOI', 0)),
                    'implied_volatility':float(opt.get('IV', 0)),
                    'expiry':            expiry_timestamp
                })
            except Exception as e:
                Logger.error(f"Option processing error: {e}")
        return processed

    def get_positions(self):
        if not self.client:
            return []
        try:
            with SuppressPrints():
                positions = self.client.positions()
            return positions if positions and isinstance(positions, list) else []
        except Exception as e:
            Logger.error(f"get_positions failed: {e}")
            return []

    def get_market_status(self) -> dict:
        result = {"is_open": False, "message": "Market is closed"}
        try:
            status_list = self.client.get_market_status()
            for exch in status_list:
                if exch.get("Exch") == "N" and exch.get("ExchType") == "C":
                    if exch.get("MarketStatus") == "Open":
                        result["is_open"] = True
                        result["message"] = "NSE Cash market is OPEN"
                    else:
                        result["message"] = "NSE Cash market is CLOSED"
                    break
        except Exception as e:
            Logger.error(f"get_market_status failed: {e}")
        return result

    def place_order_api(self, scripCode, direction, quantity, price,
                        strike_price=None, option_type=None, exchange=None):
        try:
            order_type = 'B' if direction.upper() == 'BUY' else 'S'
            self.client.place_order(
                OrderType=order_type, Exchange=exchange, ExchangeType='D',
                ScripCode=int(scripCode), Qty=quantity, Price=price,
                IsIntraday=False, DisQty=0, StopLossPrice=0,
                IsVTD=False, IOCOrder=False, IsAHPlaced=exchange
            )
            detail = f"{direction} {quantity} {scripCode}"
            if strike_price and option_type:
                detail += f" {strike_price} {option_type}"
            detail += f" @ Rs.{price:.2f}"
            print(f"ORDER PLACED: {detail}")
            return True
        except Exception as e:
            print(f"Order placement failed: {e}")
            return False

# ===============================
# TECHNICAL INDICATORS
# ===============================

class TechnicalIndicators:
    @staticmethod
    def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        try:
            df['SMA_5']       = ta.trend.sma_indicator(df["Close"], window=5)
            df['SMA_10']      = ta.trend.sma_indicator(df["Close"], window=10)
            df['SMA_20']      = ta.trend.sma_indicator(df["Close"], window=20)
            df['EMA_9']       = ta.trend.ema_indicator(df["Close"], window=9)
            df['EMA_21']      = ta.trend.ema_indicator(df["Close"], window=21)
            df['EMA_50']      = ta.trend.ema_indicator(df["Close"], window=50)
            df['VWAP']        = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df["RSI"]         = ta.momentum.rsi(df["Close"], window=14)
            df["MACD"]        = ta.trend.macd_diff(df["Close"])
            df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
            df["BB_High"]     = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
            df["BB_Low"]      = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
            df['ATR']         = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
            df['Stoch_K']     = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
            df['Stoch_D']     = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
            df['Williams_R']  = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        except Exception as e:
            Logger.error(f"Indicator compute error: {e}")
        return df

# ===============================
# CANDLESTICK PATTERN DETECTOR  (S8)
# ===============================

class CandlestickPatterns:
    @staticmethod
    def _body(c):   return abs(c['Close'] - c['Open'])
    @staticmethod
    def _range(c):  return c['High'] - c['Low']
    @staticmethod
    def _upper_wick(c): return c['High'] - max(c['Close'], c['Open'])
    @staticmethod
    def _lower_wick(c): return min(c['Close'], c['Open']) - c['Low']
    @staticmethod
    def _is_bullish(c): return c['Close'] > c['Open']
    @staticmethod
    def _is_bearish(c): return c['Close'] < c['Open']

    @staticmethod
    def hammer(c) -> bool:
        body=CandlestickPatterns._body(c); lwick=CandlestickPatterns._lower_wick(c)
        uwick=CandlestickPatterns._upper_wick(c); rng=CandlestickPatterns._range(c)
        if rng==0 or body==0: return False
        return lwick>=2*body and uwick<=0.3*body and body<=0.35*rng

    @staticmethod
    def dragonfly_doji(c) -> bool:
        body=CandlestickPatterns._body(c); lwick=CandlestickPatterns._lower_wick(c)
        rng=CandlestickPatterns._range(c)
        if rng==0: return False
        return body<=0.05*rng and lwick>=0.6*rng

    @staticmethod
    def bullish_engulfing(prev, curr) -> bool:
        return (CandlestickPatterns._is_bearish(prev) and CandlestickPatterns._is_bullish(curr) and
                curr['Open']<=prev['Close'] and curr['Close']>=prev['Open'])

    @staticmethod
    def morning_star(c1, c2, c3) -> bool:
        b1=CandlestickPatterns._body(c1); b2=CandlestickPatterns._body(c2); b3=CandlestickPatterns._body(c3)
        return (CandlestickPatterns._is_bearish(c1) and b1>0 and b2<=0.3*b1 and
                CandlestickPatterns._is_bullish(c3) and b3>=0.5*b1 and
                c3['Close']>(c1['Open']+c1['Close'])/2)

    @staticmethod
    def piercing_line(prev, curr) -> bool:
        mid=(prev['Open']+prev['Close'])/2
        return (CandlestickPatterns._is_bearish(prev) and CandlestickPatterns._is_bullish(curr) and
                curr['Open']<prev['Low'] and curr['Close']>mid and curr['Close']<prev['Open'])

    @staticmethod
    def shooting_star(c) -> bool:
        body=CandlestickPatterns._body(c); uwick=CandlestickPatterns._upper_wick(c)
        lwick=CandlestickPatterns._lower_wick(c); rng=CandlestickPatterns._range(c)
        if rng==0 or body==0: return False
        return uwick>=2*body and lwick<=0.3*body and body<=0.35*rng

    @staticmethod
    def gravestone_doji(c) -> bool:
        body=CandlestickPatterns._body(c); uwick=CandlestickPatterns._upper_wick(c)
        rng=CandlestickPatterns._range(c)
        if rng==0: return False
        return body<=0.05*rng and uwick>=0.6*rng

    @staticmethod
    def bearish_engulfing(prev, curr) -> bool:
        return (CandlestickPatterns._is_bullish(prev) and CandlestickPatterns._is_bearish(curr) and
                curr['Open']>=prev['Close'] and curr['Close']<=prev['Open'])

    @staticmethod
    def evening_star(c1, c2, c3) -> bool:
        b1=CandlestickPatterns._body(c1); b2=CandlestickPatterns._body(c2); b3=CandlestickPatterns._body(c3)
        return (CandlestickPatterns._is_bullish(c1) and b1>0 and b2<=0.3*b1 and
                CandlestickPatterns._is_bearish(c3) and b3>=0.5*b1 and
                c3['Close']<(c1['Open']+c1['Close'])/2)

    @staticmethod
    def dark_cloud_cover(prev, curr) -> bool:
        mid=(prev['Open']+prev['Close'])/2
        return (CandlestickPatterns._is_bullish(prev) and CandlestickPatterns._is_bearish(curr) and
                curr['Open']>prev['High'] and curr['Close']<mid and curr['Close']>prev['Open'])

    @staticmethod
    def detect_bullish_pattern(df: pd.DataFrame) -> Optional[str]:
        if len(df)<3: return None
        c1,c2,c3 = df.iloc[-3],df.iloc[-2],df.iloc[-1]
        if CandlestickPatterns.morning_star(c1,c2,c3):    return "MORNING_STAR"
        if CandlestickPatterns.bullish_engulfing(c2,c3):  return "BULLISH_ENGULFING"
        if CandlestickPatterns.piercing_line(c2,c3):      return "PIERCING_LINE"
        if CandlestickPatterns.hammer(c3):                return "HAMMER"
        if CandlestickPatterns.dragonfly_doji(c3):        return "DRAGONFLY_DOJI"
        return None

    @staticmethod
    def detect_bearish_pattern(df: pd.DataFrame) -> Optional[str]:
        if len(df)<3: return None
        c1,c2,c3 = df.iloc[-3],df.iloc[-2],df.iloc[-1]
        if CandlestickPatterns.evening_star(c1,c2,c3):    return "EVENING_STAR"
        if CandlestickPatterns.bearish_engulfing(c2,c3):  return "BEARISH_ENGULFING"
        if CandlestickPatterns.dark_cloud_cover(c2,c3):   return "DARK_CLOUD_COVER"
        if CandlestickPatterns.shooting_star(c3):         return "SHOOTING_STAR"
        if CandlestickPatterns.gravestone_doji(c3):       return "GRAVESTONE_DOJI"
        return None

# ===============================
# S8 SWING + TRENDLINE HELPER
# ===============================

class S8Helper:
    @staticmethod
    def get_recent_swing_highs(df, lookback=50):
        sub=df.tail(lookback+4).iloc[:-2]; highs=[]
        for i in range(2,len(sub)-2):
            c=sub.iloc[i]
            if (c['High']>sub.iloc[i-1]['High'] and c['High']>sub.iloc[i-2]['High'] and
                    c['High']>sub.iloc[i+1]['High'] and c['High']>sub.iloc[i+2]['High']):
                highs.append(c['High'])
        return highs

    @staticmethod
    def get_recent_swing_lows(df, lookback=50):
        sub=df.tail(lookback+4).iloc[:-2]; lows=[]
        for i in range(2,len(sub)-2):
            c=sub.iloc[i]
            if (c['Low']<sub.iloc[i-1]['Low'] and c['Low']<sub.iloc[i-2]['Low'] and
                    c['Low']<sub.iloc[i+1]['Low'] and c['Low']<sub.iloc[i+2]['Low']):
                lows.append(c['Low'])
        return lows

    @staticmethod
    def trendline_value_at_last(points):
        if len(points)<2: return None
        x1,y1=len(points)-2,points[-2]; x2,y2=len(points)-1,points[-1]
        slope=(y2-y1)/max(x2-x1,1)
        return y2+slope

    @staticmethod
    def is_trendline_broken_up(df, swing_highs):
        if len(swing_highs)<2: return False
        tl_val=S8Helper.trendline_value_at_last(swing_highs[-2:])
        if tl_val is None: return False
        return swing_highs[-1]<swing_highs[-2] and float(df.iloc[-1]['Close'])>tl_val

    @staticmethod
    def is_trendline_broken_down(df, swing_lows):
        if len(swing_lows)<2: return False
        tl_val=S8Helper.trendline_value_at_last(swing_lows[-2:])
        if tl_val is None: return False
        return swing_lows[-1]>swing_lows[-2] and float(df.iloc[-1]['Close'])<tl_val

    @staticmethod
    def nearest_swing_low_to_price(swing_lows, price, atr):
        candidates=[l for l in swing_lows if abs(l-price)<=atr and l<price]
        return max(candidates) if candidates else None

    @staticmethod
    def nearest_swing_high_to_price(swing_highs, price, atr):
        candidates=[h for h in swing_highs if abs(h-price)<=atr and h>price]
        return min(candidates) if candidates else None

    @staticmethod
    def evaluate(df, current_price, atr, oi_sentiment):
        result={
            "ce_signal":False,"pe_signal":False,
            "ce_pattern":None,"pe_pattern":None,
            "ce_swing":None,"pe_swing":None,
            "ce_sl":None,"pe_sl":None,
            "ce_target":None,"pe_target":None,
            "swing_highs":[],"swing_lows":[],
            "tl_break_up":False,"tl_break_down":False,
        }
        if df is None or len(df)<10: return result

        swing_highs=S8Helper.get_recent_swing_highs(df)
        swing_lows =S8Helper.get_recent_swing_lows(df)
        result["swing_highs"]=swing_highs
        result["swing_lows"] =swing_lows
        atr_safe=max(atr,1.0)

        tl_up=S8Helper.is_trendline_broken_up(df,swing_highs)
        result["tl_break_up"]=tl_up
        near_low =S8Helper.nearest_swing_low_to_price(swing_lows,current_price,atr_safe*1.5)
        bull_pat  =CandlestickPatterns.detect_bullish_pattern(df)

        if tl_up and near_low is not None and oi_sentiment=="BULLISH":
            sl   = round(near_low - atr_safe*0.5, 2)
            risk = current_price - sl
            result.update({
                "ce_signal":True,"ce_pattern":bull_pat,
                "ce_swing":near_low,"ce_sl":sl,
                "ce_target": round(current_price + risk*RR["S8"], 2)
            })

        tl_down=S8Helper.is_trendline_broken_down(df,swing_lows)
        result["tl_break_down"]=tl_down
        near_high=S8Helper.nearest_swing_high_to_price(swing_highs,current_price,atr_safe*1.5)
        bear_pat  =CandlestickPatterns.detect_bearish_pattern(df)

        if tl_down and near_high is not None and oi_sentiment=="BEARISH":
            sl   = round(near_high + atr_safe*0.5, 2)
            risk = sl - current_price
            result.update({
                "pe_signal":True,"pe_pattern":bear_pat,
                "pe_swing":near_high,"pe_sl":sl,
                "pe_target": round(current_price - risk*RR["S8"], 2)
            })

        return result

# ===============================
# MARKET ANALYZER
# ===============================

class MarketAnalyzer:
    pcr_history: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _bs_price(S, K, T, r, sigma, option_type):
        if T<=0 or sigma<=0: return 0.0
        d1=(math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)); d2=d1-sigma*math.sqrt(T)
        if option_type=="CE": return S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
        return K*math.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

    @staticmethod
    def calculate_implied_volatility(price, S, K, T, r, option_type):
        if price<=0 or T<=0: return 0.0
        sigma=0.30
        for _ in range(50):
            bs_price=MarketAnalyzer._bs_price(S,K,T,r,sigma,option_type)
            d1=(math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
            vega=S*norm.pdf(d1)*math.sqrt(T)
            if vega==0: break
            diff=bs_price-price
            if abs(diff)<1e-4: break
            sigma-=diff/vega; sigma=max(0.001,min(sigma,5))
        return float(round(sigma*100,2))

    @staticmethod
    def calculate_greeks(S, K, T, r, iv, option_type):
        if T<=0 or iv<=0: return {"delta":0,"gamma":0,"theta":0,"vega":0}
        sigma=iv/100
        d1=(math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)); d2=d1-sigma*math.sqrt(T)
        delta=norm.cdf(d1) if option_type=="CE" else -norm.cdf(-d1)
        gamma=norm.pdf(d1)/(S*sigma*math.sqrt(T))
        vega=S*norm.pdf(d1)*math.sqrt(T)/100
        if option_type=="CE":
            theta=(-S*norm.pdf(d1)*sigma/(2*math.sqrt(T))-r*K*math.exp(-r*T)*norm.cdf(d2))/365
        else:
            theta=(-S*norm.pdf(d1)*sigma/(2*math.sqrt(T))+r*K*math.exp(-r*T)*norm.cdf(-d2))/365
        return {"delta":float(round(delta,4)),"gamma":float(round(gamma,6)),
                "theta":float(round(theta,2)),"vega":float(round(vega,4))}

    @staticmethod
    def find_support_resistance_option_chain(option_chain_data, current_price, pclose_price, window=800.0):
        if not option_chain_data: return [],[],  "No data"
        calls=[o for o in option_chain_data if o.get("option_type")=="CE" and o.get("open_interest",0)>0]
        puts =[o for o in option_chain_data if o.get("option_type")=="PE" and o.get("open_interest",0)>0]
        put_cand =[o for o in puts  if pclose_price-window<=o.get("strike_price",0)<=pclose_price]
        call_cand=[o for o in calls if pclose_price<=o.get("strike_price",0)<=pclose_price+window]
        if not put_cand and not call_cand: return [],[],  "No strikes in window"
        max_put_oi =max((o.get("open_interest",0) for o in put_cand), default=0)
        max_call_oi=max((o.get("open_interest",0) for o in call_cand),default=0)
        def strength(opt,atm,smx):
            oi=float(opt.get("open_interest",0))
            if smx>0 and oi<smx*0.25: return 0.0
            chg=max(float(opt.get("change_in_oi",0)),0.0)
            vol=float(opt.get("volume",0))
            if oi>0: vol=min(vol,oi*2.0)
            dist=abs(float(opt.get("strike_price",0))-atm)
            atm_w=max(1.0,200.0/(1.0+dist))
            return oi*0.50+chg*0.35+vol*0.10+atm_w*0.55
        put_levels =sorted([(o.get("strike_price"),strength(o,current_price,max_put_oi))
                             for o in put_cand  if strength(o,current_price,max_put_oi)>0],
                            key=lambda x:x[1],reverse=True)
        call_levels=sorted([(o.get("strike_price"),strength(o,current_price,max_call_oi))
                             for o in call_cand if strength(o,current_price,max_call_oi)>0],
                            key=lambda x:x[1],reverse=True)
        supports   =[l[0] for l in put_levels[:3]]
        resistances=[l[0] for l in call_levels[:3]]
        ts=sum(l[1] for l in call_levels); tp=sum(l[1] for l in put_levels)
        if tp==0 and ts==0:  bias="Neutral"
        elif tp>=ts:         bias="🟢 Bullish Bias (Put writers dominant)"
        else:                bias="🔴 Bearish Bias (Call writers dominant)"
        return supports,resistances,bias

    @staticmethod
    def analyze_option_chain(option_chain_data, current_price, pclose_price, symbol):
        hist=MarketAnalyzer.pcr_history.setdefault(symbol,{
            "oi":[],"oi_change":[],"volume":[],"iv_history":[],
            "oi_sentiment":"NEUTRAL","oi_change_sentiment":"NEUTRAL",
            "volume_sentiment":"NEUTRAL","combined_sentiment":"NEUTRAL",
            "oi_chg_trade":None,"oi_chg_strike":None,"oi_chg_token":None,"oi_chg_entry_price":None,
            "vol_trade":None,"vol_strike":None,"vol_token":None,"vol_entry_price":None
        })
        if not option_chain_data: return {}

        def _parse_expiry(val):
            if isinstance(val,datetime): return val
            if isinstance(val,str): return datetime.strptime(val,"%Y-%m-%d")
            if isinstance(val,(int,float)): return datetime.fromtimestamp(val/1000 if val>1e12 else val)
            raise ValueError(f"Unsupported expiry: {val}")

        expiry_dates=[o.get("expiry") for o in option_chain_data if o.get("expiry")]
        if expiry_dates:
            expiry_dt=min(_parse_expiry(e) for e in expiry_dates)
            T=max((expiry_dt-datetime.now()).total_seconds(),60)/(365*24*60*60)
        else:
            T=5/365
        r=0.06

        for opt in option_chain_data:
            iv=MarketAnalyzer.calculate_implied_volatility(
                opt.get("last_price",0),current_price,opt.get("strike_price",0),T,r,opt.get("option_type"))
            greeks=MarketAnalyzer.calculate_greeks(
                current_price,opt.get("strike_price",0),T,r,iv,opt.get("option_type"))
            opt["implied_volatility"]=iv; opt.update(greeks)

        iv_vals=[o["implied_volatility"] for o in option_chain_data if o["implied_volatility"]>0]
        avg_iv=round(sum(iv_vals)/len(iv_vals),2) if iv_vals else 0
        hist["iv_history"].append(avg_iv)
        iv_min,iv_max=min(hist["iv_history"]),max(hist["iv_history"])
        iv_rank=round(((avg_iv-iv_min)/(iv_max-iv_min))*100,2) if iv_max>iv_min else 0
        iv_pct=round(sum(iv<=avg_iv for iv in hist["iv_history"])/len(hist["iv_history"])*100,2)

        gamma_by_strike={}
        for opt in option_chain_data:
            s=opt.get("strike_price"); gamma_by_strike[s]=gamma_by_strike.get(s,0)+opt.get("gamma",0)
        gamma_flip=(min(gamma_by_strike,key=lambda k:abs(k-current_price)) if gamma_by_strike else None)

        net_delta=sum(opt.get("delta",0)*opt.get("open_interest",0) for opt in option_chain_data)
        delta_bias_score=round(50+max(min(net_delta/1e6,50),-50),2)
        avg_gamma=sum(o.get("gamma",0) for o in option_chain_data)/max(len(option_chain_data),1)
        sl_points=round(max(20,current_price*avg_gamma*0.8),2)
        target_points=round(sl_points*1.8,2)

        calls=[o for o in option_chain_data if o.get("option_type")=="CE"]
        puts =[o for o in option_chain_data if o.get("option_type")=="PE"]
        call_vol=sum(o.get("volume",0) for o in calls)
        put_vol =sum(o.get("volume",0) for o in puts)
        call_oi =sum(o.get("open_interest",0) for o in calls)
        put_oi  =sum(o.get("open_interest",0) for o in puts)
        call_oi_chg=sum(o.get("change_in_oi",0) for o in calls)
        put_oi_chg =sum(o.get("change_in_oi",0) for o in puts)
        call_vp=sum(o.get("volume",0)*o.get("last_price",0) for o in calls)
        put_vp =sum(o.get("volume",0)*o.get("last_price",0) for o in puts)

        pcr_vol      =round(put_vol   /max(call_vol,1),2)
        pcr_oi       =round(put_oi    /max(call_oi,1),2)
        pcr_oi_chg   =round(put_oi_chg/max(abs(call_oi_chg),1),2)
        pcr_vol_price=round(put_vp    /max(call_vp,1),2)

        hist["oi"].append(pcr_oi); hist["oi_change"].append(pcr_oi_chg); hist["volume"].append(pcr_vol)

        if len(hist["oi"])>=2:
            prev_oi=hist["oi"][-2]; prev_chg=hist["oi_change"][-2]; prev_vol=hist["volume"][-2]
            if pcr_oi>prev_oi:        hist["oi_sentiment"]="BULLISH"
            elif pcr_oi<prev_oi:      hist["oi_sentiment"]="BEARISH"
            if pcr_oi_chg>prev_chg:   hist["oi_change_sentiment"]="BULLISH"
            elif pcr_oi_chg<prev_chg: hist["oi_change_sentiment"]="BEARISH"
            if pcr_vol>prev_vol:      hist["volume_sentiment"]="BULLISH"
            elif pcr_vol<prev_vol:    hist["volume_sentiment"]="BEARISH"
            bull=sum(s=="BULLISH" for s in [hist["oi_sentiment"],hist["oi_change_sentiment"],hist["volume_sentiment"]])
            bear=sum(s=="BEARISH" for s in [hist["oi_sentiment"],hist["oi_change_sentiment"],hist["volume_sentiment"]])
            if bull>bear:   hist["combined_sentiment"]="BULLISH"
            elif bear>bull: hist["combined_sentiment"]="BEARISH"

        return {
            'call_volume':call_vol,'put_volume':put_vol,'call_oi':call_oi,'put_oi':put_oi,
            'call_oi_change':call_oi_chg,'put_oi_change':put_oi_chg,
            'pcr_volume':pcr_vol,'pcr_oi':pcr_oi,'pcr_oi_change':pcr_oi_chg,'pcr_vol_price':pcr_vol_price,
            'oi_sentiment':hist["oi_sentiment"],'oi_change_sentiment':hist["oi_change_sentiment"],
            'volume_sentiment':hist["volume_sentiment"],'sentiment':hist["combined_sentiment"],
            'avg_iv':avg_iv,'iv_rank':iv_rank,'iv_percentile':iv_pct,
            'gamma_flip_level':gamma_flip,'delta_bias_score':delta_bias_score,
            'sl_points':sl_points,'target_points':target_points
        }

    @staticmethod
    def find_support_resistance_historical(df, current_price):
        if df.empty: return [],[]
        highs=df['High'].rolling(window=20,center=True).max()
        lows =df['Low'].rolling(window=20,center=True).min()
        pivot_highs=df[df['High']==highs]['High'].unique()
        pivot_lows =df[df['Low'] ==lows ]['Low'].unique()
        resistances=[]
        for h in pivot_highs:
            if ((df['High']>=h*0.999)&(df['Low']<=h*1.001)).sum()>=5 and h>current_price:
                resistances.append(h)
        supports=[]
        for l in pivot_lows:
            if ((df['High']>=l*0.999)&(df['Low']<=l*1.001)).sum()>=5 and l<current_price:
                supports.append(l)
        return sorted(supports,reverse=True)[:10],sorted(resistances)[:10]

    @staticmethod
    def find_swing_points(df):
        if df.empty or len(df)<5: return [],[]
        swing_highs,swing_lows=[],[]
        for i in range(4,len(df)):
            idx=i-2
            if (df.iloc[idx]['High']>df.iloc[idx-1]['High'] and df.iloc[idx]['High']>df.iloc[idx-2]['High'] and
                df.iloc[idx]['High']>df.iloc[idx+1]['High'] and df.iloc[idx]['High']>df.iloc[idx+2]['High']):
                swing_highs.append(df.iloc[idx]['High'])
            if (df.iloc[idx]['Low']<df.iloc[idx-1]['Low'] and df.iloc[idx]['Low']<df.iloc[idx-2]['Low'] and
                df.iloc[idx]['Low']<df.iloc[idx+1]['Low'] and df.iloc[idx]['Low']<df.iloc[idx+2]['Low']):
                swing_lows.append(df.iloc[idx]['Low'])
        return swing_highs,swing_lows

    @staticmethod
    def find_high_accuracy_support_resistance_combined_latest(
            historical_df, option_chain_data, current_price, pclose_price, symbol,
            tolerance_pct=0.015, min_touch_count=2):

        option_analysis=MarketAnalyzer.analyze_option_chain(option_chain_data,current_price,pclose_price,symbol)
        market_bias=option_analysis.get("sentiment","NEUTRAL")
        hist_supports,hist_resistances=MarketAnalyzer.find_support_resistance_historical(historical_df,current_price)
        swing_highs,swing_lows=MarketAnalyzer.find_swing_points(historical_df)
        oc_supports,oc_resistances,_=MarketAnalyzer.find_support_resistance_option_chain(
            option_chain_data,current_price,pclose_price)

        tol=30
        validated_supports=[]
        for lvl in oc_supports:
            lo,hi=lvl-tol,lvl+tol
            vh=[s for s in hist_supports if lo<=s<=hi]; vs=[l for l in swing_lows if lo<=l<=hi]
            if vh or vs: validated_supports.append({'level':lvl,'historical_matches':vh,'swing_matches':vs,'type':'SUPPORT'})

        validated_resistances=[]
        for lvl in oc_resistances:
            lo,hi=lvl-tol,lvl+tol
            vh=[r for r in hist_resistances if lo<=r<=hi]; vs=[h for h in swing_highs if lo<=h<=hi]
            if vh or vs: validated_resistances.append({'level':lvl,'historical_matches':vh,'swing_matches':vs,'type':'RESISTANCE'})

        call_options=[o for o in option_chain_data if o['option_type']=='CE']
        put_options =[o for o in option_chain_data if o['option_type']=='PE']
        nearest_call=(min(call_options,key=lambda x:abs(x['strike_price']-current_price)) if call_options else None)
        nearest_put =(min(put_options, key=lambda x:abs(x['strike_price']-current_price)) if put_options  else None)
        nc =({'name':nearest_call['name'],'scripCode':nearest_call['ScripCode'],
              'strike_price':nearest_call['strike_price'],'last_price':nearest_call['last_price']} if nearest_call else None)
        np_=({'name':nearest_put['name'],'scripCode':nearest_put['ScripCode'],
              'strike_price':nearest_put['strike_price'],'last_price':nearest_put['last_price']} if nearest_put else None)
        return {
            'current_price':current_price,'nearest_call':nc,'nearest_put':np_,
            'validated_supports':validated_supports,'validated_resistances':validated_resistances,
            'market_bias':market_bias,'oc_supports':oc_supports,'oc_resistances':oc_resistances,
            'option_analysis':option_analysis
        }

    @staticmethod
    def calculate_fibonacci_levels(df):
        if df.empty: return {}
        high=df['High'].max(); low=df['Low'].min(); diff=high-low
        return {'Fib_0':high,'Fib_23.6':high-0.236*diff,'Fib_38.2':high-0.382*diff,
                'Fib_50':high-0.5*diff,'Fib_61.8':high-0.618*diff,'Fib_78.6':high-0.786*diff,'Fib_100':low}

    @staticmethod
    def calculate_fibonacci_extension_levels(df):
        if df.empty: return {}
        high=df['High'].max(); low=df['Low'].min(); diff=high-low
        return {'Ext_127.2':high+0.272*diff,'Ext_138.2':high+0.382*diff,'Ext_161.8':high+0.618*diff,
                'Ext_200':high+diff,'Ext_261.8':high+1.618*diff,
                'Ext_Down_127.2':low-0.272*diff,'Ext_Down_138.2':low-0.382*diff,
                'Ext_Down_161.8':low-0.618*diff,'Ext_Down_200':low-diff,'Ext_Down_261.8':low-1.618*diff}

    @staticmethod
    def analyze_fibonacci_trend(df, current_price):
        if df.empty or len(df)<50: return {'trend':'INSUFFICIENT_DATA','strength':'WEAK','key_levels':[]}
        recent=df.tail(100)
        sh_idx=recent['High'].idxmax(); sl_idx=recent['Low'].idxmin()
        swing_high=recent['High'].max(); swing_low=recent['Low'].min(); diff=swing_high-swing_low
        is_up=sl_idx<sh_idx
        if is_up:
            fib={'fib_0':swing_high,'fib_23.6':swing_high-0.236*diff,'fib_38.2':swing_high-0.382*diff,
                 'fib_50':swing_high-0.5*diff,'fib_61.8':swing_high-0.618*diff,'fib_78.6':swing_high-0.786*diff,'fib_100':swing_low}
        else:
            fib={'fib_0':swing_low,'fib_23.6':swing_low+0.236*diff,'fib_38.2':swing_low+0.382*diff,
                 'fib_50':swing_low+0.5*diff,'fib_61.8':swing_low+0.618*diff,'fib_78.6':swing_low+0.786*diff,'fib_100':swing_high}
        rp=df['Close'].tail(20); momentum=(current_price-rp.iloc[0])/rp.iloc[0]*100
        return {'trend':'UPTREND' if is_up and current_price>fib['fib_38.2'] else 'DOWNTREND',
                'strength':'STRONG','swing_high':swing_high,'swing_low':swing_low,
                'momentum':round(momentum,2),'fib_levels':fib}

    @staticmethod
    def _determine_fibonacci_trend(current_price,fib_levels,is_uptrend_scenario,
                                   recent_prices,recent_highs,recent_lows,price_momentum,swing_high,swing_low):
        fib_23=fib_levels['fib_23.6'];fib_38=fib_levels['fib_38.2']
        fib_50=fib_levels['fib_50'];fib_61=fib_levels['fib_61.8'];fib_78=fib_levels['fib_78.6']
        signals,current_zone,key_levels=[],  "UNKNOWN",[]
        if is_uptrend_scenario:
            if current_price>fib_23:
                current_zone,trend,strength="ABOVE_23.6","STRONG_UPTREND","VERY_STRONG"
                key_levels=[fib_23,fib_38];fib_support,fib_resistance=fib_23,swing_high
                signals.append("Price holding above 23.6% - Very bullish")
            elif current_price>fib_38:
                current_zone="ABOVE_38.2";trend="UPTREND"
                strength="STRONG" if recent_lows.min()<=fib_38 else "MODERATE"
                key_levels=[fib_38,fib_50];fib_support,fib_resistance=fib_38,swing_high
            elif current_price>fib_50:
                current_zone="ABOVE_50"
                if recent_lows.min()<=fib_50 and price_momentum>0: trend,strength="UPTREND","MODERATE"
                else: trend,strength="CONSOLIDATING","WEAK"
                key_levels=[fib_50,fib_61];fib_support,fib_resistance=fib_50,fib_38
            elif current_price>fib_61:
                current_zone,trend,strength="ABOVE_61.8","WEAK_UPTREND","WEAK"
                key_levels=[fib_61,fib_78];fib_support,fib_resistance=fib_61,fib_50
            elif current_price>fib_78:
                current_zone,trend,strength="ABOVE_78.6","TREND_REVERSAL_RISK","VERY_WEAK"
                key_levels=[fib_78,swing_low];fib_support,fib_resistance=fib_78,fib_61
            else:
                current_zone,trend,strength="BELOW_78.6","DOWNTREND","STRONG"
                key_levels=[swing_low,fib_78];fib_support,fib_resistance=swing_low,fib_78
        else:
            if current_price<fib_23:
                current_zone,trend,strength="BELOW_23.6","STRONG_DOWNTREND","VERY_STRONG"
                key_levels=[fib_23,fib_38];fib_resistance,fib_support=fib_23,swing_low
            elif current_price<fib_38:
                current_zone="BELOW_38.2";trend="DOWNTREND"
                strength="STRONG" if recent_highs.max()>=fib_38 else "MODERATE"
                key_levels=[fib_38,fib_50];fib_resistance,fib_support=fib_38,swing_low
            elif current_price<fib_50:
                current_zone="BELOW_50"
                if recent_highs.max()>=fib_50 and price_momentum<0: trend,strength="DOWNTREND","MODERATE"
                else: trend,strength="CONSOLIDATING","WEAK"
                key_levels=[fib_50,fib_61];fib_resistance,fib_support=fib_50,fib_38
            elif current_price<fib_61:
                current_zone,trend,strength="BELOW_61.8","WEAK_DOWNTREND","WEAK"
                key_levels=[fib_61,fib_78];fib_resistance,fib_support=fib_61,fib_50
            elif current_price<fib_78:
                current_zone,trend,strength="BELOW_78.6","TREND_REVERSAL_RISK","VERY_WEAK"
                key_levels=[fib_78,swing_high];fib_resistance,fib_support=fib_78,fib_61
            else:
                current_zone,trend,strength="ABOVE_78.6","UPTREND","STRONG"
                key_levels=[swing_high,fib_78];fib_resistance,fib_support=swing_high,fib_78
        return {'trend':trend,'strength':strength,'current_zone':current_zone,'key_levels':key_levels,
                'fib_support':fib_support,'fib_resistance':fib_resistance,'signals':signals}

    @staticmethod
    def _calculate_trend_extensions(swing_high,swing_low,diff,trend):
        if 'UPTREND' in trend:
            return {'target_127.2':swing_high+0.272*diff,'target_138.2':swing_high+0.382*diff,
                    'target_161.8':swing_high+0.618*diff,'target_200':swing_high+diff}
        elif 'DOWNTREND' in trend:
            return {'target_127.2':swing_low-0.272*diff,'target_138.2':swing_low-0.382*diff,
                    'target_161.8':swing_low-0.618*diff,'target_200':swing_low-diff}
        return {}

    @staticmethod
    def get_trendlines(df):
        if df is None or df.empty: return [],[]
        (_,support_lines),_=trendln.calc_support_resistance(df['Low'].values,method=trendln.METHOD_NSQURED,accuracy=2)
        _,(_,resistance_lines)=trendln.calc_support_resistance(df['High'].values,method=trendln.METHOD_NSQURED,accuracy=2)
        return support_lines[-2:],resistance_lines[-2:]

    @staticmethod
    def get_line_value(line): return line[1][-1]

    @staticmethod
    def is_near_level(price,lines,threshold=15):
        for line in lines:
            if abs(price-MarketAnalyzer.get_line_value(line))<=threshold: return True
        return False

    @staticmethod
    def find_dynamic_support_resistance(option_chain_data,current_price):
        calls=[o for o in option_chain_data if o.get('option_type','').upper()=='CE']
        puts =[o for o in option_chain_data if o.get('option_type','').upper()=='PE']
        def cs(opt): return opt.get('open_interest',0)*0.6+max(opt.get('change_in_oi',0),0)*0.3+opt.get('volume',0)*0.1
        put_s =sorted([(o.get('strike_price'),cs(o)) for o in puts  if o.get('strike_price',0)<=current_price],key=lambda x:x[1],reverse=True)[:3]
        call_s=sorted([(o.get('strike_price'),cs(o)) for o in calls if o.get('strike_price',0)>=current_price],key=lambda x:x[1],reverse=True)[:3]
        return put_s,call_s

    @staticmethod
    def find_supply_demand_zones(df):
        swing_highs,swing_lows=MarketAnalyzer.find_swing_points(df)
        supply=[{'level':h,'zone_top':h*1.002,'zone_bottom':h*0.998,'strength':'MEDIUM'} for h in swing_highs[-5:]]
        demand=[{'level':l,'zone_top':l*1.002,'zone_bottom':l*0.998,'strength':'MEDIUM'} for l in swing_lows[-5:]]
        return supply,demand

# ======================================================
# ORB HELPER  — S7
# ======================================================

class ORBHelper:
    @staticmethod
    def get_or_build_orb(symbol, hist_df):
        now=datetime.now(); today_key=f"{symbol}_{now.date()}"
        existing=CONFIG.orb_state.get(today_key)
        if existing and existing.get("ready"): return existing
        if now.time()<dtime(9,31): return {"high":None,"low":None,"ready":False}
        if hist_df is None or hist_df.empty: return {"high":None,"low":None,"ready":False}
        df=hist_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            for col in ['Datetime','Date','datetime','date']:
                if col in df.columns:
                    df[col]=pd.to_datetime(df[col]); df=df.set_index(col); break
        today_str=now.strftime("%Y-%m-%d")
        try:    day_df=df[today_str]
        except: day_df=df[df.index.date==now.date()]
        if day_df.empty: return {"high":None,"low":None,"ready":False}
        orb_df=day_df.between_time("09:15","09:30")
        if orb_df.empty: return {"high":None,"low":None,"ready":False}
        orb_high=float(orb_df['High'].max()); orb_low=float(orb_df['Low'].min())
        atr_val=float(df['Close'].tail(14).std())
        orb={"high":orb_high,"low":orb_low,"atr":atr_val,"ready":True}
        CONFIG.orb_state[today_key]=orb
        Logger.info(f"[{symbol}] ORB → High:{orb_high} | Low:{orb_low}")
        return orb

# ======================================================
# TARGET CALCULATOR  — central helper for all strategies
# ======================================================

class TargetCalculator:
    """
    Computes (sl_price, target_price) for every strategy and direction.

    For option premium-based exits:
      - SL   = entry_price - risk_per_unit   (CE) / entry_price + risk_per_unit (PE)
      - Target = entry_price + risk_per_unit * RR  (CE) / entry_price - risk_per_unit * RR (PE)

    risk_per_unit is derived from the UNDERLYING index ATR converted to option premium
    using a simple approximation:  risk_per_unit ≈ atr * delta (default 0.5 for ATM)

    All S7 and S8 targets also use underlying price levels (stored separately as
    sl_price / target_price on the underlying index), but the option premium SL/target
    drives the actual trade exit.
    """

    @staticmethod
    def compute(sid: str, direction: str, entry_option_price: float,
                atr: float, current_index_price: float = 0,
                extra: Dict = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (sl_price, target_price) in option premium terms.
        extra: dict with strategy-specific data (orb levels, swing levels etc.)
        """
        if entry_option_price <= 0:
            return None, None

        extra = extra or {}
        rr    = RR.get(sid, 2.0)

        # ── ATM delta approximation (0.5 for ATM options)
        atm_delta = 0.50

        # risk in option premium ≈ ATR of underlying × delta
        # clamp to at least 5 points, at most 30% of entry price
        risk = max(atr * atm_delta, 5.0)
        risk = min(risk, entry_option_price * 0.30)

        # ── Strategy-specific overrides ────────────────────────────────
        if sid == "S1":
            # Zone bounce: tight SL — 1 × risk, target 1.5 ×
            risk = max(atr * 0.4, 4.0)

        elif sid == "S4":
            # BB breakout: wider SL for momentum
            risk = max(atr * 0.6, 6.0)

        elif sid == "S7":
            # ORB: SL is fixed to ORB boundary in premium equivalent
            orb_high = extra.get("orb_high")
            orb_low  = extra.get("orb_low")
            if direction == "CE" and orb_low and current_index_price:
                # underlying risk = current_price - orb_low
                underlying_risk = max(current_index_price - orb_low, atr * 0.5)
                risk = max(underlying_risk * atm_delta, 5.0)
            elif direction == "PE" and orb_high and current_index_price:
                underlying_risk = max(orb_high - current_index_price, atr * 0.5)
                risk = max(underlying_risk * atm_delta, 5.0)

        elif sid == "S8":
            # S8: SL is swing level based — override if already computed
            s8_sl = extra.get("ce_sl") if direction == "CE" else extra.get("pe_sl")
            if s8_sl and current_index_price:
                underlying_risk = abs(current_index_price - s8_sl)
                risk = max(underlying_risk * atm_delta, 5.0)

        # ── Final SL / Target in option premium ──────────────────────
        if direction == "CE":
            sl_price     = round(entry_option_price - risk, 2)
            target_price = round(entry_option_price + risk * rr, 2)
        else:   # PE
            sl_price     = round(entry_option_price - risk, 2)   # option price drops too
            target_price = round(entry_option_price + risk * rr, 2)

        # Clamp SL to minimum 1
        sl_price = max(sl_price, 1.0)

        return sl_price, target_price

# ===============================
# MAIN TRADING SYSTEM
# ===============================

class SimplifiedTradingSystem:
    def __init__(self, credentials_path="credentials.json"):
        self.db_manager = DatabaseManager()
        self.api_client = APIClient(credentials_path)
        self.analyzer   = MarketAnalyzer()
        if not self.api_client.initialize_client():
            raise Exception("Failed to initialize API client")

        # Restore any OPEN trades from DB so monitoring continues after restart
        self._restore_summary = self.db_manager.restore_open_trades()

    def display_positions(self):
        positions = self.api_client.get_positions()
        if not positions:
            print("\nACTIVE POSITIONS: None"); return
        total_pnl = 0.0
        print(f"\n{'='*110}")
        print(f"ACTIVE POSITIONS ({len(positions)}) - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*110}")
        print(f"{'#':<3}{'ScripName':<42}{'Code':<8}{'AvgRate':<10}{'LTP':<10}{'NetQty':<8}{'P&L':<12}{'Day%':<8}")
        print(f"{'='*110}")
        for i, pos in enumerate(positions, 1):
            name=pos.get('ScripName','')[:40]; code=pos.get('ScripCode',0)
            avg=pos.get('AvgRate',0); ltp=pos.get('LTP',0)
            qty=pos.get('NetQty',0); mtom=pos.get('MTOM',0); pclose=pos.get('PreviousClose',0)
            day_pct=((ltp-pclose)/pclose*100) if pclose else 0
            total_pnl+=mtom; pind="+" if mtom>=0 else "-"; dind="+" if day_pct>=0 else ""
            print(f"{i:<3}{name:<42}{code:<8}Rs.{avg:<7.2f}Rs.{ltp:<7.2f}{qty:<8}"
                  f"{pind}Rs.{abs(mtom):<9.2f}{dind}{day_pct:<6.2f}%")
        print(f"{'='*110}"); print(f"TOTAL P&L: Rs.{total_pnl:,.2f}"); print(f"{'='*110}")

    def display_positions_compact(self):
        positions = self.api_client.get_positions()
        if not positions:
            print("No active positions"); return
        total_pnl = sum(pos.get('MTOM',0) for pos in positions)
        print("\nACTIVE POSITIONS SUMMARY:"); print("-"*80)
        print(f"{'Scrip':40} {'Qty':>6} {'LTP':>10} {'P&L':>12}"); print("-"*80)
        for pos in positions:
            print(f"{pos.get('ScripName','')[:40]:40} {pos.get('NetQty',0):6} "
                  f"{pos.get('LTP',0):10.2f} {pos.get('MTOM',0):12.2f}")
        print("-"*80); print(f"{'Total P&L':>58}: {total_pnl:12.2f}"); print("-"*80)

    def run_analysis_loop(self):
        cycle = 0
        try:
            while True:
                if self.api_client.is_trading_time():
                    cycle += 1
                    print(f"\n{'='*60}")
                    print(f"ANALYSIS CYCLE #{cycle} — {datetime.now().strftime('%H:%M:%S')}")
                    print(f"{'='*60}")
                    for symbol in CONFIG.SYMBOLS:
                        try:    self.analyze_symbol(symbol)
                        except Exception as e: Logger.error(f"Error analysing {symbol}: {e}")
                    Logger.info(f"Cycle complete. Waiting {CONFIG.DATA_UPDATE_INTERVAL}s…")
                    time.sleep(CONFIG.DATA_UPDATE_INTERVAL)
                else:
                    Logger.warning("Market closed — waiting…")
                    time.sleep(30)
        except KeyboardInterrupt:
            Logger.info("Stopped by user.")
        except Exception as e:
            Logger.error(f"Critical error: {e}")

    def analyze_symbol(self, symbol: str):
        option_chain = self.api_client.get_option_chain_data(symbol)
        if option_chain:
            self.db_manager.store_option_chain_data(symbol, option_chain)

        current_data = self.api_client.get_current_price(symbol)
        if not current_data:
            Logger.error(f"No price data for {symbol}"); return

        current_price = current_data['ltp']
        open_price    = current_data.get('open', 0)

        historical_data = self.api_client.fetch_historical_data(symbol, mins="5m", days_back=100)
        if historical_data is None or historical_data.empty:
            Logger.error(f"No historical data for {symbol}"); return

        result = self.analyzer.find_high_accuracy_support_resistance_combined_latest(
            historical_data, option_chain, current_price, open_price, symbol)

        hist_df = TechnicalIndicators.compute_all_indicators(historical_data)

        self.display_and_trade(
            symbol, current_price,
            result['nearest_call'], result['nearest_put'],
            current_data, result['option_analysis'],
            result['oc_supports'], result['oc_resistances'],
            result['market_bias'],
            result['validated_supports'], result['validated_resistances'],
            hist_df, option_chain
        )

    # ─────────────────────────────────────────────────────────
    def display_and_trade(
            self, symbol, current_price, nearest_call, nearest_put,
            current_data, option_analysis, oc_supports, oc_resistances,
            market_bias, validated_supports, validated_resistances,
            hist_df, option_chain):

        try:
            if hist_df is None or hist_df.empty:
                print(f"❌ No historical data for {symbol}"); return

            def _f(row, col, default=0):
                v = row.get(col, default)
                return (float(v) if v is not None and
                        not (isinstance(v, float) and math.isnan(v)) else float(default))

            last = hist_df.iloc[-1]
            prev = hist_df.iloc[-2] if len(hist_df) >= 2 else last

            ema9       = _f(last, "EMA_9")
            ema21      = _f(last, "EMA_21")
            prev_ema9  = _f(prev, "EMA_9")
            prev_ema21 = _f(prev, "EMA_21")
            vwap       = _f(last, "VWAP")
            rsi        = _f(last, "RSI", 50)
            bb_high    = _f(last, "BB_High")
            bb_low     = _f(last, "BB_Low")
            macd       = _f(last, "MACD", 0)
            prev_macd  = _f(prev, "MACD", 0)
            stoch_k    = _f(last, "Stoch_K", 50)
            stoch_d    = _f(last, "Stoch_D", 50)
            atr        = _f(last, "ATR", 0)
            last_vol   = int(_f(last, "Volume"))
            avg_vol    = int(hist_df["Volume"].tail(20).mean() or 1)

            oi_sentiment = option_analysis.get("oi_sentiment", "NEUTRAL")
            delta_bias   = option_analysis.get("delta_bias_score", 50)

            print(f"\n{'='*80}")
            print(f"  {symbol}  |  Rs.{current_price:.2f}  |  {market_bias}")
            print(f"  EMA9:{ema9:.2f}  EMA21:{ema21:.2f}  RSI:{rsi:.1f}  "
                  f"MACD:{macd:.4f}  VWAP:{vwap:.2f}  ATR:{atr:.2f}  OI:{oi_sentiment}")
            print(f"{'='*80}")

            # ── HELPERS ──────────────────────────────────────────────
            def get_token(opt):
                if not opt: return None
                return opt.get("ScripCode") or opt.get("scripCode") or opt.get("token")

            def get_option_price(opt_type, token, strike):
                """Get current option last_price from live option chain."""
                for o in option_chain:
                    if o.get("option_type") == opt_type:
                        if o.get("ScripCode") == token or o.get("strike_price") == strike:
                            return float(o.get("last_price", 0))
                return None

            # ── ZONE (S1) ─────────────────────────────────────────────
            candle_low  = current_data["low"]
            candle_high = current_data["high"]
            range_val   = candle_high - candle_low
            zone        = range_val * 0.25
            ce_zone     = candle_low  <= current_price <= (candle_low  + zone)
            pe_zone     = (candle_high - zone) <= current_price <= candle_high

            # ── VALIDATED S/R ─────────────────────────────────────────
            def _near_validated_support(price, tol=40):
                for s in validated_supports:
                    for m in s.get("historical_matches", [s.get("level", 0)]):
                        if abs(price - m) <= tol: return True
                return False

            def _near_validated_resistance(price, tol=40):
                for r in validated_resistances:
                    for m in r.get("historical_matches", [r.get("level", 0)]):
                        if abs(price - m) <= tol: return True
                return False

            bullish_touch = any(
                candle_low < m < current_price
                for s in validated_supports for m in s.get("historical_matches", [])
            )
            bearish_touch = any(
                candle_high > m > current_price
                for r in validated_resistances for m in r.get("historical_matches", [])
            )

            # ── ORB (S7) ──────────────────────────────────────────────
            orb       = ORBHelper.get_or_build_orb(symbol, hist_df)
            orb_ready = orb.get("ready", False)
            orb_high  = orb.get("high")
            orb_low   = orb.get("low")
            orb_atr   = orb.get("atr", atr)
            orb_ce_cond = (orb_ready and orb_high is not None and
                           current_price > orb_high and oi_sentiment == "BULLISH")
            orb_pe_cond = (orb_ready and orb_low  is not None and
                           current_price < orb_low  and oi_sentiment == "BEARISH")
            if orb_ready and orb_high:
                print(f"  ORB → High:{orb_high} | Low:{orb_low}")

            # ── S8 ────────────────────────────────────────────────────
            s8 = S8Helper.evaluate(hist_df, current_price, atr, oi_sentiment)
            if s8["ce_signal"] or s8["pe_signal"]:
                print(f"  S8 → TL_UP:{s8['tl_break_up']} TL_DN:{s8['tl_break_down']} "
                      f"| Bull:{s8['ce_pattern']} Swing_Low:{s8['ce_swing']} "
                      f"| Bear:{s8['pe_pattern']} Swing_High:{s8['pe_swing']}")

            # ── STATE INIT ────────────────────────────────────────────
            ALL_STRATEGIES = ["S1","S2","S3","S4","S5","S6","S7","S8"]
            if symbol not in CONFIG.oi_state:
                CONFIG.oi_state[symbol] = {
                    "bullish": False, "bearish": False,
                    "trades": {s: {"CE": new_trade_state(), "PE": new_trade_state()}
                               for s in ALL_STRATEGIES}
                }
            for s in ALL_STRATEGIES:
                if s not in CONFIG.oi_state[symbol]["trades"]:
                    CONFIG.oi_state[symbol]["trades"][s] = {
                        "CE": new_trade_state(), "PE": new_trade_state()
                    }

            state = CONFIG.oi_state[symbol]
            state["bullish"] = bullish_touch
            state["bearish"] = bearish_touch

            # ============================================================
            # STRATEGIES — ENTRY CONDITIONS
            # ============================================================
            strategies = {
                # S1 · Zone Bounce
                "S1": lambda d: (
                    (d=="CE" and ce_zone and state["bullish"]) or
                    (d=="PE" and pe_zone and state["bearish"])
                ),
                # S2 · EMA Crossover
                "S2": lambda d: (
                    (d=="CE" and prev_ema9<=prev_ema21 and ema9>ema21 and oi_sentiment=="BULLISH") or
                    (d=="PE" and prev_ema9>=prev_ema21 and ema9<ema21 and oi_sentiment=="BEARISH")
                ),
                # S3 · VWAP + RSI
                "S3": lambda d: (
                    (d=="CE" and current_price>vwap and rsi>55 and oi_sentiment=="BULLISH") or
                    (d=="PE" and current_price<vwap and rsi<45 and oi_sentiment=="BEARISH")
                ),
                # S4 · BB Breakout + Volume
                "S4": lambda d: (
                    (d=="CE" and current_price>bb_high and last_vol>avg_vol*1.5 and oi_sentiment=="BULLISH") or
                    (d=="PE" and current_price<bb_low  and last_vol>avg_vol*1.5 and oi_sentiment=="BEARISH")
                ),
                # S5 · RSI Extreme at S/R
                "S5": lambda d: (
                    (d=="CE" and rsi<35 and _near_validated_support(current_price)    and oi_sentiment=="BULLISH") or
                    (d=="PE" and rsi>65 and _near_validated_resistance(current_price) and oi_sentiment=="BEARISH")
                ),
                # S6 · MACD + Stoch
                "S6": lambda d: (
                    (d=="CE" and prev_macd<=0 and macd>0 and stoch_k>stoch_d and oi_sentiment=="BULLISH") or
                    (d=="PE" and prev_macd>=0 and macd<0 and stoch_k<stoch_d and oi_sentiment=="BEARISH")
                ),
                # S7 · ORB
                "S7": lambda d: (
                    (d=="CE" and orb_ce_cond) or
                    (d=="PE" and orb_pe_cond)
                ),
                # S8 · Swing + Trendline + Candle
                "S8": lambda d: (
                    (d=="CE" and s8["ce_signal"]) or
                    (d=="PE" and s8["pe_signal"])
                ),
            }

            # ============================================================
            # PER-STRATEGY EXIT CONDITIONS
            # Returns (should_exit: bool, reason: str)
            # ============================================================
            def exit_condition(sid: str, opt_type: str, trade: Dict):
                entry_p    = trade.get("entry_price")
                sl_p       = trade.get("sl_price")
                target_p   = trade.get("target_price")
                cur_opt_p  = get_option_price(opt_type, trade.get("token"), trade.get("strike"))

                # ── Universal SL / Target hit check (option premium) ──
                if cur_opt_p is not None and entry_p is not None:
                    if sl_p     is not None and cur_opt_p <= sl_p:
                        return True, "SL_HIT"
                    if target_p is not None and cur_opt_p >= target_p:
                        return True, "TARGET_HIT"

                # ── Strategy-specific signal reversal exits ────────────
                if sid == "S1":
                    rev = not ce_zone if opt_type=="CE" else not pe_zone
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S2":
                    rev = ema9 < ema21 if opt_type=="CE" else ema9 > ema21
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S3":
                    rev = current_price < vwap if opt_type=="CE" else current_price > vwap
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S4":
                    rev = current_price < bb_high if opt_type=="CE" else current_price > bb_low
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S5":
                    rev = rsi > 50 if opt_type=="CE" else rsi < 50
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S6":
                    rev = macd < 0 if opt_type=="CE" else macd > 0
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S7":
                    # Exit if price re-enters ORB range (underlying level check)
                    tl = trade.get("orb_low")
                    th = trade.get("orb_high")
                    if opt_type == "CE":
                        rev = (tl is not None and current_price < tl)
                    else:
                        rev = (th is not None and current_price > th)
                    return rev, "SIGNAL_REVERSAL"

                elif sid == "S8":
                    # Additional check: underlying price beyond swing level
                    s8_sl = trade.get("sl_price")   # this is the option premium SL
                    # We already caught it in the universal check above
                    return False, ""

                return not strategies[sid](opt_type), "SIGNAL_REVERSAL"

            # ============================================================
            # FORCE-EXIT NEAR MARKET CLOSE
            # ============================================================
            force_exit = self.api_client.is_force_exit_time()

            # ============================================================
            # ENTRY LOOP
            # ============================================================
            for sid, cond in strategies.items():
                rr_ratio = RR.get(sid, 2.0)

                # ── CE ENTRY ─────────────────────────────────────────
                if nearest_call:
                    token = get_token(nearest_call)
                    if token is not None:
                        ce = state["trades"][sid]["CE"]
                        if not ce["active"] and cond("CE"):
                            entry_price = nearest_call.get("last_price", 0)
                            qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots

                            # Build strategy-specific extra data
                            extra_data: Dict = {}
                            if sid == "S7":
                                extra_data = {"orb_high": orb_high, "orb_low": orb_low}
                            elif sid == "S8":
                                extra_data = {
                                    "swing_level": s8["ce_swing"],
                                    "pattern":     s8["ce_pattern"],
                                    "ce_sl":        s8["ce_sl"],
                                }

                            # Compute SL and Target
                            sl_price, target_price = TargetCalculator.compute(
                                sid, "CE", entry_price, atr,
                                current_price, extra_data
                            )

                            trade_id = self.db_manager.insert_trade(
                                symbol=symbol, strategy=sid, option_type="CE",
                                strike=nearest_call.get("strike_price"),
                                token=token, qty=qty, entry_price=entry_price,
                                entry_oi=oi_sentiment, entry_delta=delta_bias,
                                orb_high=extra_data.get("orb_high"),
                                orb_low =extra_data.get("orb_low"),
                                sl_price=sl_price,
                                target_price=target_price,
                                swing_level=extra_data.get("swing_level"),
                                pattern=extra_data.get("pattern"),
                            )
                            ce.update({
                                "active": True, "trade_id": trade_id,
                                "token": token, "symbol": nearest_call.get("name"),
                                "entry_price": entry_price, "qty": qty,
                                "sl_price": sl_price, "target_price": target_price,
                                "orb_high": extra_data.get("orb_high"),
                                "orb_low":  extra_data.get("orb_low"),
                                "swing_level": extra_data.get("swing_level"),
                                "pattern":     extra_data.get("pattern"),
                            })
                            print(f"  ✅ {sid} CE ENTRY | Strike:{nearest_call.get('strike_price')} "
                                  f"| Prem:{entry_price:.2f} | SL:{sl_price} | TGT:{target_price} "
                                  f"| RR:1:{rr_ratio} | RSI:{rsi:.1f}")

                # ── PE ENTRY ─────────────────────────────────────────
                if nearest_put:
                    token = get_token(nearest_put)
                    if token is not None:
                        pe = state["trades"][sid]["PE"]
                        if not pe["active"] and cond("PE"):
                            entry_price = nearest_put.get("last_price", 0)
                            qty = CONFIG.LOT_SIZES[symbol.upper()] * CONFIG.number_of_lots

                            extra_data = {}
                            if sid == "S7":
                                extra_data = {"orb_high": orb_high, "orb_low": orb_low}
                            elif sid == "S8":
                                extra_data = {
                                    "swing_level": s8["pe_swing"],
                                    "pattern":     s8["pe_pattern"],
                                    "pe_sl":        s8["pe_sl"],
                                }

                            sl_price, target_price = TargetCalculator.compute(
                                sid, "PE", entry_price, atr,
                                current_price, extra_data
                            )

                            trade_id = self.db_manager.insert_trade(
                                symbol=symbol, strategy=sid, option_type="PE",
                                strike=nearest_put.get("strike_price"),
                                token=token, qty=qty, entry_price=entry_price,
                                entry_oi=oi_sentiment, entry_delta=delta_bias,
                                orb_high=extra_data.get("orb_high"),
                                orb_low =extra_data.get("orb_low"),
                                sl_price=sl_price,
                                target_price=target_price,
                                swing_level=extra_data.get("swing_level"),
                                pattern=extra_data.get("pattern"),
                            )
                            pe.update({
                                "active": True, "trade_id": trade_id,
                                "token": token, "symbol": nearest_put.get("name"),
                                "entry_price": entry_price, "qty": qty,
                                "sl_price": sl_price, "target_price": target_price,
                                "orb_high": extra_data.get("orb_high"),
                                "orb_low":  extra_data.get("orb_low"),
                                "swing_level": extra_data.get("swing_level"),
                                "pattern":     extra_data.get("pattern"),
                            })
                            print(f"  ✅ {sid} PE ENTRY | Strike:{nearest_put.get('strike_price')} "
                                  f"| Prem:{entry_price:.2f} | SL:{sl_price} | TGT:{target_price} "
                                  f"| RR:1:{rr_ratio} | RSI:{rsi:.1f}")

            # ============================================================
            # EXIT LOOP
            # ============================================================
            for sid, sides in state["trades"].items():
                for opt_type, trade in sides.items():
                    if not trade["active"]:
                        continue

                    exit_p = get_option_price(opt_type, trade.get("token"), trade.get("strike"))
                    if exit_p is None or trade.get("entry_price") is None or trade.get("qty") is None:
                        continue

                    pnl = (exit_p - trade["entry_price"]) * trade["qty"]

                    if force_exit:
                        should_exit, reason = True, "FORCE_CLOSE_EOD"
                    else:
                        should_exit, reason = exit_condition(sid, opt_type, trade)

                    if should_exit:
                        self.db_manager.close_trade(trade["trade_id"], exit_p, pnl, reason)
                        sl_p  = trade.get("sl_price")
                        tgt_p = trade.get("target_price")
                        print(f"  ❌ {sid} {opt_type} EXIT [{reason}] | "
                              f"Entry:{trade['entry_price']:.2f} | Exit:{exit_p:.2f} | "
                              f"SL:{sl_p} | TGT:{tgt_p} | PnL:{pnl:.2f}")
                        trade.update(new_trade_state())

        except Exception as e:
            Logger.error(f"display_and_trade error for {symbol}: {e}")
            import traceback
            traceback.print_exc()

# ===============================
# MAIN
# ===============================

# ===============================
# MAIN (FIXED)
# ===============================

import asyncio

async def main():
    print("\n" + "="*80)
    print("  PROFESSIONAL TRADING SYSTEM v5.0")
    print("  S1-Zone | S2-EMA | S3-VWAP | S4-BB | S5-RSI | S6-MACD | S7-ORB | S8-Swing")
    print("  TARGETS: SL + Target on every strategy (option premium based)")
    print("="*80)

    try:
        system = SimplifiedTradingSystem()
        system.run_analysis_loop()   # your existing loop
    except Exception as e:
        Logger.error(f"Startup error: {e}")
        await asyncio.sleep(5)  # prevent crash loop


if __name__ == "__main__":
    asyncio.run(main())
