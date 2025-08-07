from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, date
import pandas as pd
import numpy as np

@dataclass
class OptionContract:
    """Represents a single option contract with all necessary pricing and risk data."""
    
    # Contract details
    symbol: str
    strike: float
    expiration: date
    option_type: str  # 'call' or 'put'
    underlying_price: float
    
    # Market data
    market_premium: float
    implied_volatility: float
    
    # Greeks
    delta: float
    gamma: float
    vega: float
    theta: Optional[float] = None
    
    # Contract specifications
    contract_multiplier: int = 100  # Shares per contract (typically 100 for equity options)
    
    # Computed fields
    theoretical_price: Optional[float] = None
    expected_return: Optional[float] = None  # market_premium - theoretical_price
    
    def __post_init__(self):
        """Validate the option contract data."""
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.market_premium < 0:
            raise ValueError("market_premium cannot be negative")
        if self.implied_volatility <= 0:
            raise ValueError("implied_volatility must be positive")

@dataclass
class Position:
    """Represents a position in an option contract."""
    
    contract: OptionContract
    quantity: int  # Positive for long, negative for short
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def notional_exposure(self) -> float:
        """Total notional exposure of the position."""
        return abs(self.quantity) * self.contract.contract_multiplier * self.contract.underlying_price
    
    @property
    def premium_cost(self) -> float:
        """Net premium cost (positive for debit, negative for credit)."""
        return self.quantity * self.contract.market_premium * self.contract.contract_multiplier
    
    @property
    def expected_pnl(self) -> float:
        """Expected P&L from this position."""
        if self.contract.expected_return is None:
            return 0.0
        # expected_return already includes contract_multiplier from pricing.py
        return self.quantity * self.contract.expected_return

@dataclass
class Portfolio:
    """Represents a portfolio of option positions."""
    
    positions: List[Position]
    
    def __post_init__(self):
        if not self.positions:
            self.positions = []
    
    @property
    def total_delta(self) -> float:
        """Total portfolio delta."""
        return sum(pos.quantity * pos.contract.delta for pos in self.positions)
    
    @property
    def total_gamma(self) -> float:
        """Total portfolio gamma."""
        return sum(pos.quantity * pos.contract.gamma for pos in self.positions)
    
    @property
    def total_vega(self) -> float:
        """Total portfolio vega."""
        return sum(pos.quantity * pos.contract.vega for pos in self.positions)
    
    @property
    def net_premium(self) -> float:
        """Net premium paid (positive) or received (negative)."""
        return sum(pos.premium_cost for pos in self.positions)
    
    @property
    def expected_return(self) -> float:
        """Total expected return from the portfolio."""
        return sum(pos.expected_pnl for pos in self.positions)
    
    @property
    def capital_used(self) -> float:
        """Capital used (only counting long positions' premiums)."""
        return sum(max(0, pos.premium_cost) for pos in self.positions)
    
    def margin_used(self, margin_per_contract: float = 2000.0) -> float:
        """Total margin used (based on short positions)."""
        short_contracts = sum(abs(pos.quantity) for pos in self.positions if pos.is_short)
        return short_contracts * margin_per_contract
    
    def add_position(self, contract: OptionContract, quantity: int):
        """Add a position to the portfolio."""
        self.positions.append(Position(contract=contract, quantity=quantity))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to DataFrame for display."""
        data = []
        for pos in self.positions:
            data.append({
                'Symbol': pos.contract.symbol,
                'Type': pos.contract.option_type,
                'Strike': pos.contract.strike,
                'Expiration': pos.contract.expiration,
                'Quantity': pos.quantity,
                'Position Type': 'Long' if pos.is_long else 'Short',
                'Market Premium': pos.contract.market_premium,
                'Delta': pos.contract.delta,
                'Gamma': pos.contract.gamma,
                'Premium Cost': pos.premium_cost,
                'Expected P&L': pos.expected_pnl
            })
        return pd.DataFrame(data)

@dataclass
class OptimizationConstraints:
    """Configuration for portfolio optimization constraints."""
    
    # Neutrality constraints
    delta_tolerance: float = 0.5
    gamma_tolerance: float = 0.2
    
    # Capital constraints
    capital_budget: float = 500_000
    margin_cap: float = 100_000
    margin_per_contract: float = 2_000
    
    # Risk constraints
    cvar_confidence: float = 0.95
    cvar_limit: float = -75_000  # Maximum acceptable loss in worst case scenarios
    
    # Position limits
    max_contracts_per_option: int = 50
    min_contracts_per_option: int = -50

@dataclass
class OptimizationResult:
    """Results from portfolio optimization."""
    
    portfolio: Portfolio
    objective_value: float
    optimization_status: str
    constraints_satisfied: bool
    
    # Risk metrics
    portfolio_cvar: float
    portfolio_std: float
    
    # Constraint violations (if any)
    delta_violation: float = 0.0
    gamma_violation: float = 0.0
    capital_violation: float = 0.0
    margin_violation: float = 0.0
    cvar_violation: float = 0.0 