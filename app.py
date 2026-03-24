"""
TradeSys v5.0 — Flask Dashboard
Serves UI + data APIs from the SQLite DB written by win.py.
Run: python app.py
"""

import json
import sqlite3
import os
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
DB_PATH = os.environ.get("DB_PATH", "trading_data.db")

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_list(rows):
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Trades (open / closed / today) ──────────────────────
@app.route("/api/trades")
def api_trades():
    symbol = request.args.get("symbol", "").upper()
    status = request.args.get("status", "all").upper()  # OPEN | CLOSED | ALL
    today  = date.today().isoformat()

    try:
        with get_db() as conn:
            if status == "OPEN":
                rows = conn.execute(
                    "SELECT * FROM option_trades WHERE status='OPEN' ORDER BY entry_time DESC"
                ).fetchall()
            elif status == "CLOSED":
                rows = conn.execute(
                    "SELECT * FROM option_trades WHERE status='CLOSED' AND date(entry_time)=? ORDER BY exit_time DESC",
                    (today,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM option_trades ORDER BY entry_time DESC LIMIT 200"
                ).fetchall()
        return jsonify(rows_to_list(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Today P&L summary ────────────────────────────────────
@app.route("/api/summary")
def api_summary():
    today = date.today().isoformat()
    try:
        with get_db() as conn:
            row = conn.execute("""
                SELECT
                  COUNT(*) as total_trades,
                  SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END) as open_trades,
                  SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? THEN 1 ELSE 0 END) as closed_today,
                  SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? THEN pnl ELSE 0 END) as pnl_today,
                  SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? AND pnl>0 THEN 1 ELSE 0 END) as wins
                FROM option_trades
            """, (today, today, today)).fetchone()
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Option chain (latest snapshot for symbol) ────────────
@app.route("/api/option_chain")
def api_option_chain():
    symbol = request.args.get("symbol", "NIFTY").upper()
    try:
        with get_db() as conn:
            # Get the latest timestamp for this symbol
            ts_row = conn.execute(
                "SELECT MAX(timestamp) as ts FROM option_chain_data WHERE symbol=?",
                (symbol,)
            ).fetchone()
            if not ts_row or not ts_row["ts"]:
                return jsonify([])
            latest_ts = ts_row["ts"]
            rows = conn.execute(
                """SELECT strike_price, option_type, last_price, volume,
                          open_interest, change_in_oi, implied_volatility
                   FROM option_chain_data
                   WHERE symbol=? AND timestamp=?
                   ORDER BY strike_price ASC""",
                (symbol, latest_ts)
            ).fetchall()
        return jsonify(rows_to_list(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 5-min candle data (last 200 candles) ─────────────────
@app.route("/api/chart")
def api_chart():
    symbol = request.args.get("symbol", "NIFTY").upper()
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT timestamp, open_price as open, high, low, close, volume
                   FROM market_data
                   WHERE symbol=?
                   ORDER BY timestamp ASC
                   LIMIT 200""",
                (symbol,)
            ).fetchall()

        candles = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["timestamp"])
                # LightweightCharts expects Unix timestamp (seconds) or "time" key
                candles.append({
                    "time":  int(ts.timestamp()),
                    "open":  float(r["open"]  or 0),
                    "high":  float(r["high"]  or 0),
                    "low":   float(r["low"]   or 0),
                    "close": float(r["close"] or 0),
                })
            except Exception:
                continue
        return jsonify(candles)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Latest market summary for a symbol ───────────────────
@app.route("/api/market_summary")
def api_market_summary():
    symbol = request.args.get("symbol", "NIFTY").upper()
    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT * FROM market_analysis_summary
                   WHERE symbol=?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol,)
            ).fetchone()
        if row:
            return jsonify(dict(row))
        # Fall back to latest market_data
        with get_db() as conn:
            row2 = conn.execute(
                """SELECT open_price as open, high, low, close, volume
                   FROM market_data WHERE symbol=? ORDER BY timestamp DESC LIMIT 1""",
                (symbol,)
            ).fetchone()
        if row2:
            d = dict(row2)
            d["current_price"] = d.get("close", 0)
            return jsonify(d)
        return jsonify({"current_price": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Strategy performance breakdown ───────────────────────
@app.route("/api/strategy_stats")
def api_strategy_stats():
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT strategy,
                  COUNT(*) as total,
                  SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
                  SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END) as losses,
                  ROUND(SUM(pnl),2) as total_pnl,
                  ROUND(AVG(pnl),2) as avg_pnl
                FROM option_trades
                WHERE status='CLOSED'
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """).fetchall()
        return jsonify(rows_to_list(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Health check ─────────────────────────────────────────
@app.route("/api/health")
def api_health():
    db_exists = os.path.exists(DB_PATH)
    return jsonify({
        "status": "ok",
        "db": DB_PATH,
        "db_exists": db_exists,
        "time": datetime.now().isoformat()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n{'='*60}")
    print(f"  TradeSys v5.0 Dashboard")
    print(f"  http://localhost:{port}")
    print(f"  DB: {DB_PATH}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)