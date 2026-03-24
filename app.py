"""
TradeSys v5.0 — Flask Dashboard (TURSO VERSION)
"""

import os
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request
from turso_db import get_db

app = Flask(__name__)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def rows_to_list(rows):
    return [dict(r) for r in rows]


def safe_rows(result):
    try:
        if result is None:
            return []

        # Turso normal
        if hasattr(result, "rows"):
            return result.rows or []

        # fallback
        if isinstance(result, list):
            return result

        # edge case
        if hasattr(result, "results"):
            return result.results

        return []

    except Exception as e:
        print("safe_rows error:", e)
        return []


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Trades ───────────────────────────────
@app.route("/api/trades")
def api_trades():
    status = request.args.get("status", "ALL").upper()
    today  = date.today().isoformat()

    try:
        db = get_db()

        if status == "OPEN":
            result = db.execute(
                "SELECT * FROM option_trades WHERE status='OPEN' ORDER BY entry_time DESC"
            )
        elif status == "CLOSED":
            result = db.execute(
                "SELECT * FROM option_trades WHERE status='CLOSED' AND date(entry_time)=? ORDER BY exit_time DESC",
                [today]
            )
        else:
            result = db.execute(
                "SELECT * FROM option_trades ORDER BY entry_time DESC LIMIT 200"
            )

        rows = safe_rows(result)
        return jsonify(rows_to_list(rows))

    except Exception as e:
        print("TRADES ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── Summary ──────────────────────────────
@app.route("/api/summary")
def api_summary():
    today = date.today().isoformat()

    try:
        db = get_db()

        result = db.execute("""
            SELECT
              COUNT(*) as total_trades,
              SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END) as open_trades,
              SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? THEN 1 ELSE 0 END) as closed_today,
              SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? THEN pnl ELSE 0 END) as pnl_today,
              SUM(CASE WHEN status='CLOSED' AND date(entry_time)=? AND pnl>0 THEN 1 ELSE 0 END) as wins
            FROM option_trades
        """, [today, today, today])

        rows = safe_rows(result)

        if not rows:
            return jsonify({})

        return jsonify(dict(rows[0]))

    except Exception as e:
        print("SUMMARY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── Option Chain ─────────────────────────
@app.route("/api/option_chain")
def api_option_chain():
    symbol = request.args.get("symbol", "NIFTY").upper()

    try:
        db = get_db()

        ts_result = db.execute(
            "SELECT MAX(timestamp) as ts FROM option_chain_data WHERE symbol=?",
            [symbol]
        )

        ts_rows = safe_rows(ts_result)

        if not ts_rows or not ts_rows[0].get("ts"):
            return jsonify([])

        latest_ts = ts_rows[0]["ts"]

        result = db.execute("""
            SELECT strike_price, option_type, last_price, volume,
                   open_interest, change_in_oi, implied_volatility
            FROM option_chain_data
            WHERE symbol=? AND timestamp=?
            ORDER BY strike_price ASC
        """, [symbol, latest_ts])

        rows = safe_rows(result)
        return jsonify(rows_to_list(rows))

    except Exception as e:
        print("OPTION CHAIN ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── Chart Data ───────────────────────────
@app.route("/api/chart")
def api_chart():
    symbol = request.args.get("symbol", "NIFTY").upper()

    try:
        db = get_db()

        result = db.execute("""
            SELECT timestamp, open_price as open, high, low, close, volume
            FROM market_data
            WHERE symbol=?
            ORDER BY timestamp ASC
            LIMIT 200
        """, [symbol])

        rows = safe_rows(result)

        candles = []

        for r in rows:
            try:
                ts = datetime.fromisoformat(r["timestamp"])
                candles.append({
                    "time": int(ts.timestamp()),
                    "open": float(r["open"] or 0),
                    "high": float(r["high"] or 0),
                    "low": float(r["low"] or 0),
                    "close": float(r["close"] or 0),
                })
            except:
                continue

        return jsonify(candles)

    except Exception as e:
        print("CHART ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── Market Summary ───────────────────────
@app.route("/api/market_summary")
def api_market_summary():
    symbol = request.args.get("symbol", "NIFTY").upper()

    try:
        db = get_db()

        result = db.execute("""
            SELECT * FROM market_analysis_summary
            WHERE symbol=?
            ORDER BY timestamp DESC LIMIT 1
        """, [symbol])

        rows = safe_rows(result)

        if rows:
            return jsonify(dict(rows[0]))

        fallback = db.execute("""
            SELECT open_price as open, high, low, close, volume
            FROM market_data
            WHERE symbol=?
            ORDER BY timestamp DESC LIMIT 1
        """, [symbol])

        f_rows = safe_rows(fallback)

        if f_rows:
            d = dict(f_rows[0])
            d["current_price"] = d.get("close", 0)
            return jsonify(d)

        return jsonify({"current_price": 0})

    except Exception as e:
        print("MARKET SUMMARY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── Strategy Stats ───────────────────────
@app.route("/api/strategy_stats")
def api_strategy_stats():
    try:
        db = get_db()

        result = db.execute("""
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

        rows = safe_rows(result)
        return jsonify(rows_to_list(rows))

    except Exception as e:
        print("STRATEGY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── Health ───────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "db": "Turso",
        "time": datetime.now().isoformat()
    })


# ── RUN ─────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
