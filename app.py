from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/api/trades")
def trades():
    return [{"symbol": "NIFTY", "pnl": 1200}]
