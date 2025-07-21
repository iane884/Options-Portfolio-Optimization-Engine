# Options Portfolio Optimizer

A comprehensive options portfolio optimization system that maximizes expected return while maintaining delta and gamma neutrality within user-defined risk constraints.

## Overview

This tool constructs optimal options portfolios by:

- **Maximizing expected return** using premium vs. theoretical fair value analysis
- **Maintaining market neutrality** through delta ≈ 0 and gamma ≈ 0 constraints  
- **Managing risk** via CVaR (Conditional Value at Risk) limits
- **Respecting capital and margin constraints**

The system uses a single underlying's full options chain, making it ideal for hedging strategies and market-neutral trading approaches.

## Key Features

### 🎯 Optimization Algorithms
- **MILP (Mixed Integer Linear Programming)**: Exact solutions using CVXPY
- **Greedy Heuristic**: Fast approximate solutions for large option chains
- **Comparative Analysis**: Run both methods and select the best result

### 📊 Risk Management  
- **Delta-Gamma Neutrality**: Configurable tolerance bands (default ±0.5 delta, ±0.2 gamma)
- **CVaR Risk Limiting**: Normal approximation for tail risk assessment
- **Capital Controls**: Budget limits and margin requirements
- **Scenario Analysis**: P&L sensitivity to price and volatility changes

### 💡 Expected Return Model
- **Theoretical Pricing**: Black-Scholes fair value calculation
- **Mispricing Detection**: Expected return = Market Premium - Theoretical Price
- **Volatility Risk Premium**: Captures systematic option overpricing

### 🖥️ User Interface
- **Streamlit Web App**: Interactive configuration and visualization
- **Multiple Data Sources**: Predefined scenarios, test chains, CSV upload (planned)
- **Export Capabilities**: Portfolio compositions to CSV format

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd options-optimizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation**:
   ```bash
   python test_optimizer.py
   ```

## Quick Start

### Command Line Testing
```bash
# Run the test suite to verify everything works
python test_optimizer.py
```

### Web Interface
```bash
# Launch the Streamlit app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Programmatic Usage
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

### Core Components

1. **`models.py`**: Data structures for options, portfolios, and constraints
2. **`pricing.py`**: Black-Scholes pricing and Greeks calculation  
3. **`risk_calculator.py`**: CVaR computation using normal approximation
4. **`data_generator.py`**: Mock options chain generation with realistic pricing
5. **`optimizers.py`**: MILP and greedy optimization algorithms
6. **`portfolio_optimizer.py`**: Main orchestrator class
7. **`app.py`**: Streamlit web interface

### Optimization Problem Formulation

**Objective**: Maximize Σ(Expected Return × Quantity)

**Subject to**:
- **Delta Neutrality**: |Σ(Delta × Quantity)| ≤ δ_tolerance
- **Gamma Neutrality**: |Σ(Gamma × Quantity)| ≤ γ_tolerance  
- **Capital Limit**: Σ(Long Premium Costs) ≤ Capital Budget
- **Margin Limit**: Σ(Short Contracts) × Margin/Contract ≤ Margin Cap
- **CVaR Constraint**: Portfolio CVaR ≥ CVaR Limit
- **Position Bounds**: Min ≤ Quantity ≤ Max (per contract)

## Configuration

### Default Constraints
- **Delta Tolerance**: ±0.5 (very low directional exposure)
- **Gamma Tolerance**: ±0.2 (stable delta as underlying moves)
- **Capital Budget**: $500,000
- **Margin Cap**: $100,000  
- **Margin per Contract**: $2,000 (flat rate approximation)
- **CVaR Limit**: -$75,000 (95% confidence)

### Market Scenarios
- **Balanced**: 20% volatility, moderate skew
- **High Vol**: 35% volatility, increased skew  
- **Low Vol**: 12% volatility, minimal skew
- **Skewed**: Strong put volatility premium

## Example Output

```
Portfolio Optimization Results
============================
Expected Return: $15,250
Portfolio Delta: 0.02
Portfolio Gamma: -0.05
Constraints Satisfied: ✅

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

### Expected Return Calculation
The system calculates expected returns as the difference between market premiums and Black-Scholes theoretical prices:

```
Expected Return = Market Premium - BS_Price(S, K, T, r, σ_implied)
```

Positive values indicate overpriced options (profitable to short), while negative values suggest underpriced options (profitable to buy).

### CVaR Risk Model
Uses a delta-gamma-normal approximation:

1. **Portfolio P&L Mean**: Σ(Expected Returns)
2. **Portfolio P&L Std**: √(Delta Risk² + Position Risk²)  
3. **CVaR Calculation**: μ - σ × φ(z_α)/α (closed-form normal)

### Solver Selection
- **MILP**: Tries CPLEX → GUROBI → CBC → GLPK in order
- **Fallback**: Uses default CVXPY solver if commercial solvers unavailable
- **Fractional Support**: Optional relaxation of integer constraints

## Limitations & Future Enhancements

### Current Limitations
- Single underlying asset only
- Normal approximation for risk (simplified)
- Flat margin rate (not Greeks-based)
- No transaction costs
- Static optimization (no rebalancing)

### Planned Enhancements
- Multi-asset portfolio support
- Monte Carlo risk simulation
- Dynamic margin calculations
- Real-time data integration
- Advanced Greeks management (charm, vanna, etc.)
- Backtesting framework

## Contributing

This is an MVP focused on core backend functionality. The web interface prioritizes functionality over aesthetics as requested.

Key areas for contribution:
- Enhanced risk models
- Additional optimization algorithms  
- Real market data connectors
- Advanced visualization features
- Performance optimizations

## License

This project is provided as-is for educational and research purposes. 