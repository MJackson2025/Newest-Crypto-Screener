<details>
<summary>Klicke zum Ausklappen</summary>

import yfinance as yf
import pandas as pd
import ta
import streamlit as st
import openai

st.set_page_config(page_title="KI Crypto Screener", layout="wide")
st.title("🚀 KI-basierter Krypto-Screener – 8 Coins")
st.markdown("Technische Analyse + GPT-Bewertung für BTC, ETH, SOL, ADA, AVAX, MATIC, LINK, ATOM")

# API-Key aus Streamlit Secrets
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("🔐 OpenAI API-Key fehlt. Bitte in Settings > Secrets eintragen.")
    st.stop()

crypto_tickers = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Cosmos (ATOM)": "ATOM-USD"
}

@st.cache_data
def analyze_crypto(ticker):
    data = yf.Ticker(ticker).history(period="90d", interval="1d")
    data.dropna(inplace=True)
    if len(data) < 50:
        raise ValueError("Nicht genügend Daten für technische Analyse")
    data["rsi"] = ta.momentum.RSIIndicator(close=data["Close"]).rsi()
    data["sma_50"] = ta.trend.SMAIndicator(close=data["Close"], window=50).sma_indicator()
    data["sma_200"] = ta.trend.SMAIndicator(close=data["Close"], window=200).sma_indicator()
    data["vol_sma20"] = data["Volume"].rolling(20).mean()
    latest = data.iloc[-1]
    signals = {
        "Preis": round(latest["Close"], 2),
        "RSI": round(latest["rsi"], 2),
        "RSI <40": latest["rsi"] < 40,
        "50er >200er": latest["sma_50"] > latest["sma_200"],
        "Vol >20-Tage": latest["Volume"] > 1.2 * latest["vol_sma20"]
    }
    signals["Kaufen"] = signals["RSI <40"] and signals["50er >200er"] and signals["Vol >20-Tage"]
    return signals

def gpt_analysis(name, signals):
    prompt = (
        f"Bewerte {name} anhand dieser Daten:\n"
        f"- RSI: {signals['RSI']}\n"
        f"- 50er >200er ? {signals['50er >200er']}\n"
        f"- RSI <40 ? {signals['RSI <40']}\n"
        f"- Volumen über 20-Tage-Schnitt? {signals['Vol >20-Tage']}\n"
        "Kurze, präzise Einschätzung auf Deutsch."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler GPT: {e}"

results, gpt_texts = {}, {}

for name, ticker in crypto_tickers.items():
    try:
        s = analyze_crypto(ticker)
        results[name] = s
        gpt_texts[name] = gpt_analysis(name, s)
    except Exception as e:
        results[name] = {"Fehler": str(e)}
        gpt_texts[name] = "-"

df = pd.DataFrame(results).T
st.dataframe(df)

st.download_button("📥 CSV export", df.to_csv(index=True).encode(), "crypto_screen_results.csv")

st.markdown("## GPT Einschätzung")
for name in crypto_tickers:
    st.subheader(name)
    st.write(gpt_texts.get(name, "-"))
