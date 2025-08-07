# Options Portfolio Optimizer

A tool that finds the best options trades to maximize profit while keeping your portfolio balanced and within your risk limits.

## What it does

- Finds overpriced/underpriced options using Black-Scholes pricing
- Builds a portfolio that's neutral to market moves (delta/gamma balanced)
- Respects your capital budget and risk limits
- Shows you exactly what to buy/sell and how much money you can expect to make

## Quick Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test that everything works:**
   ```bash
   python test_optimizer.py
   ```

3. **Launch the web interface:**
   ```bash
   streamlit run app.py
   ```
   Then open your browser to `http://localhost:8501`

## How to use

1. **Load data** - Choose a market scenario (balanced, high volatility, etc.)
2. **Set your limits** - Capital budget, risk tolerance, delta/gamma limits
3. **Run optimization** - Click "Run Optimization" to find the best trades
4. **Review results** - See your expected profit, risk metrics, and specific trades

## Example Output

```
Expected Return: $50,342
Portfolio Delta: -0.499 (neutral)
Portfolio Gamma: 0.199 (stable)
Sharpe Ratio: 1.327

Positions:
- Short SPY_450_C_250906: 25 contracts
- Long SPY_445_P_250906: 15 contracts
- Short SPY_455_P_250906: 10 contracts

Risk Metrics:
- CVaR: -$27,927 (max loss in worst case)
- Capital Used: $365,453
- Margin Used: $100,000
```

## Configuration

**Default Settings:**
- Capital Budget: $500,000
- Delta Tolerance: ±0.5 (very neutral)
- Gamma Tolerance: ±0.2 (stable)
- CVaR Limit: -$75,000 (risk limit)
- Margin Cap: $100,000

You can adjust these in the web interface sidebar.

## Files

- `app.py` - Web interface
- `portfolio_optimizer.py` - Main optimization logic
- `test_optimizer.py` - Test script
- `requirements.txt` - Python dependencies