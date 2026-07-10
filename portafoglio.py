import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Impostazioni grafici
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# Le azioni che vogliamo analizzare (ticker Yahoo Finance)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'BRK-B']
# Apple, Microsoft, Google, Amazon, JPMorgan, Johnson&Johnson, ExxonMobil, Berkshire Hathaway
# Diversi settori: tech, finanza, salute, energia — ottimo per la diversificazione!

inizio = '2019-01-01'
fine   = '2024-01-01'

print("Scarico i dati storici...")
dati_raw = yf.download(tickers, start=inizio, end=fine)

# Prendiamo solo il prezzo di chiusura giornaliero
prezzi = dati_raw['Close'].dropna()

print(f"Dati scaricati: {prezzi.shape[0]} giorni di trading, {prezzi.shape[1]} azioni")
print(prezzi.tail())  # Mostra le ultime 5 righe per controllo







# Rendimenti logaritmici giornalieri
# log(P_t / P_{t-1}) per ogni giorno e ogni azione
rendimenti = np.log(prezzi / prezzi.shift(1)).dropna()

print("\nStatistiche dei rendimenti giornalieri:")
print(rendimenti.describe())

# Visualizziamo i rendimenti 
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    axes[i].plot(rendimenti[ticker], alpha=0.7, color='royalblue', linewidth=0.8)
    axes[i].set_title(f'Rendimenti giornalieri {ticker}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Data', fontsize=9)
    axes[i].set_ylabel('Rendimento logaritmico', fontsize=9)
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

plt.suptitle('Rendimenti giornalieri — Tutte le azioni', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('rendimenti_tutti.png', dpi=150)
plt.show()






# Numero di giorni di trading in un anno (converti da giornaliero ad annuale)
TRADING_DAYS = 252

# Rendimento medio annualizzato di ogni azione
rendimenti_medi = rendimenti.mean() * TRADING_DAYS

# Matrice di covarianza annualizzata
# Misura come le azioni si muovono insieme (fondamentale per Markowitz)
cov_matrix = rendimenti.cov() * TRADING_DAYS

# Tasso risk-free: rendimento BTP italiani 10 anni circa
RISK_FREE = 0.03  # 3% annuo

def calcola_rendimento(pesi):
    """Rendimento atteso annuale del portafoglio"""
    return np.dot(pesi, rendimenti_medi)

def calcola_volatilita(pesi):
    """Volatilità (rischio) annuale del portafoglio"""
    # Formula matriciale di Markowitz: sqrt(w^T * Σ * w)
    return np.sqrt(np.dot(pesi.T, np.dot(cov_matrix, pesi)))

def calcola_sharpe(pesi):
    """Sharpe Ratio del portafoglio"""
    return (calcola_rendimento(pesi) - RISK_FREE) / calcola_volatilita(pesi)

# Test: portafoglio equipesato (stessa % su ogni azione)
n = len(tickers)
pesi_uguali = np.array([1/n] * n)
print(f"\nPortafoglio equipesato:")
print(f"  Rendimento atteso: {calcola_rendimento(pesi_uguali):.2%}")
print(f"  Volatilità:        {calcola_volatilita(pesi_uguali):.2%}")
print(f"  Sharpe Ratio:      {calcola_sharpe(pesi_uguali):.3f}")






print("\nSimulo portafogli casuali per costruire la frontiera efficiente...")

N_PORTAFOGLI = 10000
risultati = np.zeros((3, N_PORTAFOGLI))  # [rendimento, volatilità, sharpe]
pesi_salvati = []

for i in range(N_PORTAFOGLI):
    # Genera pesi casuali che sommano a 1
    pesi = np.random.random(n)
    pesi = pesi / pesi.sum()
    
    risultati[0, i] = calcola_rendimento(pesi)
    risultati[1, i] = calcola_volatilita(pesi)
    risultati[2, i] = calcola_sharpe(pesi)
    pesi_salvati.append(pesi)

# Trova il portafoglio con il massimo Sharpe Ratio
idx_max_sharpe = np.argmax(risultati[2])
pesi_ottimali = pesi_salvati[idx_max_sharpe]






print("Ottimizzazione formale con scipy...")

# scipy.minimize minimizza, quindi minimizzo il negativo dello Sharpe
def neg_sharpe(pesi):
    return -calcola_sharpe(pesi)

# Vincolo: i pesi devono sommare a 1
vincoli = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

# Bounds: ogni peso tra 0% e 100% (no posizioni short)
bounds = tuple((0, 1) for _ in range(n))

# Punto di partenza: portafoglio equipesato
x0 = pesi_uguali

ottimizzazione = minimize(neg_sharpe, x0, method='SLSQP',
                          bounds=bounds, constraints=vincoli)

pesi_ottimali = ottimizzazione.x

print("\n✅ PORTAFOGLIO OTTIMALE (max Sharpe Ratio):")
for ticker, peso in zip(tickers, pesi_ottimali):
    if peso > 0.001:  # Mostra solo pesi significativi
        print(f"  {ticker:8s}: {peso:.1%}")
print(f"\n  Rendimento atteso: {calcola_rendimento(pesi_ottimali):.2%}")
print(f"  Volatilità:        {calcola_volatilita(pesi_ottimali):.2%}")
print(f"  Sharpe Ratio:      {calcola_sharpe(pesi_ottimali):.3f}")






fig, ax = plt.subplots(figsize=(12, 8))

# Nuvola di portafogli casuali, colorati per Sharpe Ratio
scatter = ax.scatter(
    risultati[1, :] * 100,  # volatilità in %
    risultati[0, :] * 100,  # rendimento in %
    c=risultati[2, :],      # colore = Sharpe Ratio
    cmap='viridis',
    alpha=0.5,
    s=5
)
plt.colorbar(scatter, label='Sharpe Ratio')

# Portafoglio ottimale
ax.scatter(
    calcola_volatilita(pesi_ottimali) * 100,
    calcola_rendimento(pesi_ottimali) * 100,
    color='red', marker='*', s=400, zorder=5, label='Portafoglio Ottimale'
)

# Portafoglio equipesato
ax.scatter(
    calcola_volatilita(pesi_uguali) * 100,
    calcola_rendimento(pesi_uguali) * 100,
    color='orange', marker='D', s=150, zorder=5, label='Portafoglio Equipesato'
)

ax.set_xlabel('Rischio / Volatilità Annuale (%)', fontsize=13)
ax.set_ylabel('Rendimento Atteso Annuale (%)', fontsize=13)
ax.set_title('Frontiera Efficiente di Markowitz', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('frontiera_efficiente.png', dpi=150)
plt.show()



print("\nSimulazione Monte Carlo in corso...")

CAPITALE_INIZIALE = 10000   # €
N_SIMULAZIONI    = 1000
N_GIORNI         = 252      # 1 anno di trading

# Parametri del portafoglio ottimale
mu    = calcola_rendimento(pesi_ottimali) / TRADING_DAYS  # rendimento giornaliero
sigma = calcola_volatilita(pesi_ottimali) / np.sqrt(TRADING_DAYS)  # vol giornaliera

# Matrice per salvare tutte le simulazioni
simulazioni = np.zeros((N_GIORNI, N_SIMULAZIONI))
simulazioni[0] = CAPITALE_INIZIALE

for t in range(1, N_GIORNI):
    # Estrai N_SIMULAZIONI numeri casuali dalla distribuzione normale
    epsilon = np.random.standard_normal(N_SIMULAZIONI)
    # Applica il Moto Browniano Geometrico
    simulazioni[t] = simulazioni[t-1] * np.exp(
        (mu - 0.5 * sigma**2) + sigma * epsilon
    )

# Analisi dei risultati finali
valori_finali = simulazioni[-1]
print(f"\n📊 Risultati Monte Carlo dopo 1 anno ({N_SIMULAZIONI} simulazioni):")
print(f"  Capitale iniziale:    €{CAPITALE_INIZIALE:,.0f}")
print(f"  Mediana:              €{np.median(valori_finali):,.0f}")
print(f"  Scenario ottimistico  (95° percentile): €{np.percentile(valori_finali, 95):,.0f}")
print(f"  Scenario pessimistico  (5° percentile): €{np.percentile(valori_finali, 5):,.0f}")
print(f"  Probabilità di perdita: {(valori_finali < CAPITALE_INIZIALE).mean():.1%}")















fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Grafico sinistro: traiettorie ---
ax1.plot(simulazioni[:, :200], alpha=0.05, color='steelblue', linewidth=0.8)
ax1.plot(np.median(simulazioni, axis=1), color='red',
         linewidth=2.5, label='Mediana', zorder=5)
ax1.plot(np.percentile(simulazioni, 95, axis=1), color='green',
         linewidth=2, linestyle='--', label='95° percentile')
ax1.plot(np.percentile(simulazioni, 5, axis=1), color='orange',
         linewidth=2, linestyle='--', label='5° percentile')
ax1.axhline(y=CAPITALE_INIZIALE, color='black',
            linestyle=':', linewidth=1.5, label='Capitale iniziale')
ax1.set_title('Simulazione Monte Carlo — Traiettorie', fontsize=13, fontweight='bold')
ax1.set_xlabel('Giorni di trading')
ax1.set_ylabel('Valore del portafoglio (€)')
ax1.legend()
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x:,.0f}'))

# --- Grafico destro: distribuzione dei valori finali ---
ax2.hist(valori_finali, bins=60, color='steelblue', edgecolor='white',
         alpha=0.8, density=True)
ax2.axvline(CAPITALE_INIZIALE, color='black', linestyle=':', linewidth=2,
            label='Capitale iniziale')
ax2.axvline(np.median(valori_finali), color='red', linewidth=2,
            label=f'Mediana: €{np.median(valori_finali):,.0f}')
ax2.axvline(np.percentile(valori_finali, 5), color='orange', linewidth=2,
            linestyle='--', label=f'5°: €{np.percentile(valori_finali, 5):,.0f}')
ax2.axvline(np.percentile(valori_finali, 95), color='green', linewidth=2,
            linestyle='--', label=f'95°: €{np.percentile(valori_finali, 95):,.0f}')
ax2.set_title('Distribuzione dei valori finali dopo 1 anno', fontsize=13, fontweight='bold')
ax2.set_xlabel('Valore finale del portafoglio (€)')
ax2.set_ylabel('Densità di probabilità')
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x:,.0f}'))

plt.tight_layout()
plt.savefig('monte_carlo.png', dpi=150)
plt.show()

print("\n✅ Progetto completato! Grafici salvati.")