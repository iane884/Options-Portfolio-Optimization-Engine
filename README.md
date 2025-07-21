# Options Portfolio Optimizer

A portfolio optimization system that maximizes expected return while maintaining delta and gamma neutrality within user-defined risk constraints.

## Overview

This tool constructs optimal options portfolios by maximizing expected return using premium vs. theoretical fair value analysis while maintaining market neutrality through delta and gamma constraints, managing risk via CVaR limits, and respecting capital and margin constraints.

The system focuses on a single underlying's full options chain, making it suitable for hedging strategies and market-neutral trading approaches.

## Features

**Optimization Algorithms**
- MILP (Mixed Integer Linear Programming): Exact solutions using CVXPY
- Greedy Heuristic: Fast approximate solutions for large option chains
- Comparative Analysis: Run both methods and select the best result

**Risk Management**
- Delta-Gamma Neutrality: Configurable tolerance bands (default ±0.5 delta, ±0.2 gamma)
- CVaR Risk Limiting: Normal approximation for tail risk assessment
- Capital Controls: Budget limits and margin requirements
- Scenario Analysis: P&L sensitivity to price and volatility changes

**Expected Return Model**
- Theoretical Pricing: Black-Scholes fair value calculation
- Mispricing Detection: Expected return = Market Premium - Theoretical Price
- Volatility Risk Premium: Captures systematic option overpricing

**User Interface**
- Streamlit Web App: Interactive configuration and visualization
- Multiple Data Sources: Predefined scenarios, test chains
- Export Capabilities: Portfolio compositions to CSV format

## Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies: `pip install -r requirements.txt`
3. Test the installation: `python test_optimizer.py`

## Usage

**Command Line Testing**
```bash
python test_optimizer.py
```

**Web Interface**
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`

**Programmatic Usage**
```python
from portfolio_optimizer import PortfolioOptimizer
from models import OptimizationConstraints

# Initialize with custom constraints
constraints = OptimizationConstraints(
    delta_tolerance=0.5,      # ±0.5 delta neutrality
    gamma_tolerance=0.2,      # ±0.2 gamma neutrality  
    capital_budget=500_000,   # $500K capital limit
    margin_cap=100_000,       # $100K margin limit
    cvar_limit=-75_000        # Max $75K CVaR loss
)

optimizer = PortfolioOptimizer(constraints)

# Load market data
num_contracts, description = optimizer.load_contracts_from_scenario("balanced")

# Run optimization
result = optimizer.optimize(method="milp")

# Analyze results
summary = optimizer.get_portfolio_summary()
print(f"Expected Return: ${summary['portfolio_metrics']['Expected Return']:,.0f}")
print(f"Portfolio Delta: {summary['portfolio_metrics']['Portfolio Delta']:.3f}")
```

## Architecture

**Core Components**
- `models.py`: Data structures for options, portfolios, and constraints
- `pricing.py`: Black-Scholes pricing and Greeks calculation  
- `risk_calculator.py`: CVaR computation using normal approximation
- `data_generator.py`: Mock options chain generation with realistic pricing
- `optimizers.py`: MILP and greedy optimization algorithms
- `portfolio_optimizer.py`: Main orchestrator class
- `app.py`: Streamlit web interface

**Optimization Problem**

Objective: Maximize Σ(Expected Return × Quantity)

Subject to:
- Delta Neutrality: |Σ(Delta × Quantity)| ≤ δ_tolerance
- Gamma Neutrality: |Σ(Gamma × Quantity)| ≤ γ_tolerance  
- Capital Limit: Σ(Long Premium Costs) ≤ Capital Budget
- Margin Limit: Σ(Short Contracts) × Margin/Contract ≤ Margin Cap
- CVaR Constraint: Portfolio CVaR ≥ CVaR Limit
- Position Bounds: Min ≤ Quantity ≤ Max (per contract)

## Configuration

**Default Constraints**
- Delta Tolerance: ±0.5 (very low directional exposure)
- Gamma Tolerance: ±0.2 (stable delta as underlying moves)
- Capital Budget: $500,000
- Margin Cap: $100,000  
- Margin per Contract: $2,000 (flat rate approximation)
- CVaR Limit: -$75,000 (95% confidence)

**Market Scenarios**
- Balanced: 20% volatility, moderate skew
- High Vol: 35% volatility, increased skew  
- Low Vol: 12% volatility, minimal skew
- Skewed: Strong put volatility premium

## Example Output

```
Portfolio Optimization Results
Expected Return: $15,250
Portfolio Delta: 0.02
Portfolio Gamma: -0.05
Constraints Satisfied: Yes

Positions:
- Short SPY_450_C_240115: 5 contracts
- Long SPY_445_P_240115: 3 contracts  
- Short SPY_455_P_240215: 2 contracts

Risk Metrics:
- CVaR (95%): -$45,200
- Portfolio Volatility: $28,500
- Capital Used: $125,000
- Margin Used: $35,000
```

## Technical Details

**Expected Return Calculation**
The system calculates expected returns as the difference between market premiums and Black-Scholes theoretical prices:

```
Expected Return = Market Premium - BS_Price(S, K, T, r, σ_implied)
```

Positive values indicate overpriced options (profitable to short), while negative values suggest underpriced options (profitable to buy).

**CVaR Risk Model**
Uses a delta-gamma-normal approximation:
1. Portfolio P&L Mean: Σ(Expected Returns)
2. Portfolio P&L Std: √(Delta Risk² + Position Risk²)  
3. CVaR Calculation: μ - σ × φ(z_α)/α (closed-form normal)

**Solver Selection**
- MILP: Tries CPLEX → GUROBI → CBC → GLPK in order
- Fallback: Uses default CVXPY solver if commercial solvers unavailable
- Fractional Support: Optional relaxation of integer constraints

## Limitations

- Single underlying asset only
- Normal approximation for risk (simplified)
- Flat margin rate (not Greeks-based)
- No transaction costs
- Static optimization (no rebalancing)
