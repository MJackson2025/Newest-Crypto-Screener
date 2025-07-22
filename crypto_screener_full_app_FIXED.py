
import yfinance as yf
import pandas as pd
import ta
import streamlit as st
import openai

st.set_page_config(page_title="KI Crypto Screener", layout="wide")
st.title("🚀 KI-basierter Krypto-Screener – 8 Coins")
st.markdown("Technische Analyse + GPT-Bewertung für BTC, ETH, SOL, ADA, AVAX, MATIC, LINK, ATOM")

openai.api_key = st.secrets["OPENAI_API_KEY"]

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
    if data.empty or len(data) < 50:
        raise ValueError("Nicht genügend Daten für technische Analyse")

    data["rsi"] = ta.momentum.RSIIndicator(close=data["Close"]).rsi()
    data["sma_50"] = ta.trend.SMAIndicator(close=data["Close"], window=50).sma_indicator()
    data["sma_200"] = ta.trend.SMAIndicator(close=data["Close"], window=200).sma_indicator()
    data["volume_sma_20"] = data["Volume"].rolling(window=20).mean()

    latest = data.iloc[-1]
    signals = {
        "Kurs (USD)": round(latest["Close"], 2),
        "RSI": round(latest["rsi"], 2),
        "RSI < 40": latest["rsi"] < 40,
        "SMA50 > SMA200": latest["sma_50"] > latest["sma_200"],
        "Volumen > 20-Tage-Schnitt": latest["Volume"] > 1.2 * latest["volume_sma_20"]
    }
    signals["Kauf-Kandidat"] = all([
        signals["RSI < 40"],
        signals["SMA50 > SMA200"],
        signals["Volumen > 20-Tage-Schnitt"]
    ])
    return signals

def gpt_analysis(name, signals):
    prompt = (
        f"Du bist ein erfahrener Krypto-Trader. Bewerte die Coin {name} anhand dieser technischen Daten:\n"
        f"- RSI: {signals['RSI']}\n"
        f"- SMA50 > SMA200: {signals['SMA50 > SMA200']}\n"
        f"- RSI < 40: {signals['RSI < 40']}\n"
        f"- Volumen > Schnitt: {signals['Volumen > 20-Tage-Schnitt']}\n"
        "Gib eine kurze, präzise Einschätzung auf Deutsch ab (2–4 Sätze), ob Kauf sinnvoll ist oder nicht."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei GPT: {e}"

results = {}
gpt_texts = {}

for name, ticker in crypto_tickers.items():
    try:
        signals = analyze_crypto(ticker)
        results[name] = signals
        gpt_texts[name] = gpt_analysis(name, signals)
    except Exception as e:
        results[name] = {"Fehler": str(e)}
        gpt_texts[name] = "Keine GPT-Auswertung möglich."

df = pd.DataFrame(results).T
st.dataframe(df)

st.download_button(
    label="📥 Ergebnisse als CSV herunterladen",
    data=df.to_csv().encode('utf-8'),
    file_name="crypto_screening_results.csv",
    mime="text/csv"
)

st.markdown("## 🤖 GPT-Bewertung")
for name in crypto_tickers:
    st.subheader(name)
    st.write(gpt_texts.get(name, "Keine Daten."))
