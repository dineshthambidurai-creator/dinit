"""
TradeSys v5.0 — Flask Dashboard (TURSO VERSION)
"""

import os
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request
from turso_db import execute_query

app = Flask(__name__)

# ─────────────────────────────────────────
# HOME
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────
# TRADES
# ─────────────────────────────────────────

@app.route("/api/trades")
def api_trades():
    status = request.args.get("status", "ALL").upper()
    today = date.today().isoformat()

    try:
        if status == "OPEN":
            rows = execute_query(
                "SELECT * FROM option_trades WHERE status='OPEN' ORDER BY entry_time DESC"
            )

        elif status == "CLOSED":
            rows = execute_query(
                "SELECT * FROM option_trades WHERE status='CLOSED' AND date(entry_time)=? ORDER BY exit_time DESC",
                [today]
            )

        else:
            rows = execute_query(
                "SELECT * FROM option_trades ORDER BY entry_time DESC LIMIT 200"
            )

        return jsonify(rows)

    except Exception as e:
        print("TRADES ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    today = date.today().isoformat()

    try:
        rows = execute_query("""
            SELECT
              COUNT(*) as total_trades,
              SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END) as open_trades,
              SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? THEN 1 ELSE 0 END) as closed_today,
              SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? THEN pnl ELSE 0 END) as pnl_today,
              SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? AND pnl>0 THEN 1 ELSE 0 END) as wins
            FROM option_trades
        """, [today, today, today])

        if not rows:
            return jsonify({})

        return jsonify(rows[0])

    except Exception as e:
        print("SUMMARY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# OPTION CHAIN
# ─────────────────────────────────────────

@app.route("/api/option_chain")
def api_option_chain():
    symbol = request.args.get("symbol", "NIFTY").upper()

    try:
        ts_rows = execute_query(
            "SELECT MAX(timestamp) as ts FROM option_chain_data WHERE symbol=?",
            [symbol]
        )

        if not ts_rows or not ts_rows[0].get("ts"):
            return jsonify([])

        latest_ts = ts_rows[0]["ts"]

        rows = execute_query("""
            SELECT strike_price, option_type, last_price, volume,
                   open_interest, change_in_oi, implied_volatility
            FROM option_chain_data
            WHERE symbol=? AND timestamp=?
            ORDER BY strike_price ASC
        """, [symbol, latest_ts])

        return jsonify(rows)

    except Exception as e:
        print("OPTION CHAIN ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# CHART DATA
# ─────────────────────────────────────────

@app.route("/api/chart")
def api_chart():
    symbol = request.args.get("symbol", "NIFTY").upper()

    try:
        rows = execute_query("""
            SELECT timestamp, open_price as open, high, low, close, volume
            FROM market_data
            WHERE symbol=?
            ORDER BY timestamp ASC
            LIMIT 200
        """, [symbol])

        candles = []

        for r in rows:
            try:
                ts = datetime.fromisoformat(r["timestamp"])
                candles.append({
                    "time": int(ts.timestamp()),
                    "open": float(r.get("open", 0)),
                    "high": float(r.get("high", 0)),
                    "low": float(r.get("low", 0)),
                    "close": float(r.get("close", 0)),
                })
            except:
                continue

        return jsonify(candles)

    except Exception as e:
        print("CHART ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# MARKET SUMMARY
# ─────────────────────────────────────────

@app.route("/api/market_summary")
def api_market_summary():
    symbol = request.args.get("symbol", "NIFTY").upper()

    try:
        rows = execute_query("""
            SELECT * FROM market_analysis_summary
            WHERE symbol=?
            ORDER BY timestamp DESC LIMIT 1
        """, [symbol])

        if rows:
            return jsonify(rows[0])

        fallback = execute_query("""
            SELECT open_price as open, high, low, close, volume
            FROM market_data
            WHERE symbol=?
            ORDER BY timestamp DESC LIMIT 1
        """, [symbol])

        if fallback:
            d = fallback[0]
            d["current_price"] = d.get("close", 0)
            return jsonify(d)

        return jsonify({"current_price": 0})

    except Exception as e:
        print("MARKET ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# STRATEGY STATS
# ─────────────────────────────────────────

@app.route("/api/strategy_stats")
def api_strategy_stats():
    try:
        rows = execute_query("""
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
        """)

        return jsonify(rows)

    except Exception as e:
        print("STRATEGY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "db": "Turso",
        "time": datetime.now().isoformat()
    })


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "="*50)
    print("🚀 TradeSys Turso Dashboard Running")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=port)
