import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List
from models import OptionContract
from pricing import calculate_greeks, time_to_expiry_years

def generate_mock_options_chain(
    underlying_symbol: str = "SPY",
    underlying_price: float = 450.0,
    strike_range: tuple = (400, 500),
    strike_increment: float = 5.0,
    expiration_days: List[int] = [30, 60, 90],
    base_implied_vol: float = 0.20,
    vol_skew: float = 0.02,  # Additional vol for OTM puts
    risk_free_rate: float = 0.05
) -> List[OptionContract]:
    """
    Generate a mock options chain with realistic Greeks and pricing.
    
    Args:
        underlying_symbol: Symbol of underlying asset
        underlying_price: Current price of underlying
        strike_range: (min_strike, max_strike) tuple
        strike_increment: Spacing between strikes
        expiration_days: List of days to expiration
        base_implied_vol: Base implied volatility
        vol_skew: Additional volatility for OTM puts (volatility skew)
        risk_free_rate: Risk-free rate for Greeks calculation
    
    Returns:
        List of OptionContract objects
    """
    contracts = []
    current_date = date.today()
    
    # Generate strikes
    min_strike, max_strike = strike_range
    strikes = np.arange(min_strike, max_strike + strike_increment, strike_increment)
    
    for days in expiration_days:
        expiration = current_date + timedelta(days=days)
        time_to_expiry = time_to_expiry_years(expiration, current_date)
        
        for strike in strikes:
            # Calculate moneyness for volatility skew
            moneyness = underlying_price / strike
            
            for option_type in ['call', 'put']:
                # Apply volatility skew (higher vol for OTM puts)
                if option_type == 'put' and strike < underlying_price:
                    # OTM put - add volatility skew
                    skew_factor = (underlying_price - strike) / underlying_price * vol_skew
                    implied_vol = base_implied_vol + skew_factor
                elif option_type == 'call' and strike > underlying_price:
                    # OTM call - slight vol increase
                    skew_factor = (strike - underlying_price) / underlying_price * (vol_skew * 0.5)
                    implied_vol = base_implied_vol + skew_factor
                else:
                    # ATM or ITM options
                    implied_vol = base_implied_vol
                
                # Add some random noise to implied volatility
                vol_noise = np.random.normal(0, 0.01)  # 1% vol noise
                implied_vol = max(0.05, implied_vol + vol_noise)  # Min 5% vol
                
                # Calculate Greeks
                greeks = calculate_greeks(
                    underlying_price=underlying_price,
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=implied_vol,
                    option_type=option_type
                )
                
                # Calculate theoretical price for market premium
                from pricing import black_scholes_price
                theoretical_price = black_scholes_price(
                    underlying_price=underlying_price,
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=implied_vol,
                    option_type=option_type
                )
                
                # Add bid-ask spread and market noise to create "market" premium
                # Typical bid-ask spread is 2-5% of option value
                spread_factor = np.random.uniform(0.98, 1.02)  # ±2% from theoretical
                market_premium = max(0.01, theoretical_price * spread_factor)  # Min $0.01
                
                # Create contract
                contract = OptionContract(
                    symbol=f"{underlying_symbol}_{strike}_{option_type[0].upper()}_{expiration.strftime('%y%m%d')}",
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    underlying_price=underlying_price,
                    market_premium=market_premium,
                    implied_volatility=implied_vol,
                    delta=greeks['delta'],
                    gamma=greeks['gamma'],
                    vega=greeks['vega'],
                    theta=greeks['theta']
                )
                
                contracts.append(contract)
    
    return contracts

def generate_options_with_mispricing(
    base_contracts: List[OptionContract],
    mispricing_probability: float = 0.3,
    mispricing_magnitude: float = 0.10
) -> List[OptionContract]:
    """
    Add intentional mispricing to some options to create optimization opportunities.
    
    Args:
        base_contracts: Base list of option contracts
        mispricing_probability: Probability that any given option is mispriced
        mispricing_magnitude: Magnitude of mispricing as fraction of premium
    
    Returns:
        List of contracts with some mispriced options
    """
    mispriced_contracts = []
    
    for contract in base_contracts:
        # Randomly decide if this option should be mispriced
        if np.random.random() < mispricing_probability:
            # Create mispricing (positive = overpriced, negative = underpriced)
            mispricing_factor = np.random.uniform(-mispricing_magnitude, mispricing_magnitude)
            
            # Adjust market premium
            adjusted_premium = contract.market_premium * (1 + mispricing_factor)
            adjusted_premium = max(0.01, adjusted_premium)  # Ensure positive premium
            
            # Create new contract with adjusted premium
            mispriced_contract = OptionContract(
                symbol=contract.symbol,
                strike=contract.strike,
                expiration=contract.expiration,
                option_type=contract.option_type,
                underlying_price=contract.underlying_price,
                market_premium=adjusted_premium,
                implied_volatility=contract.implied_volatility,
                delta=contract.delta,
                gamma=contract.gamma,
                vega=contract.vega,
                theta=contract.theta,
                contract_multiplier=contract.contract_multiplier
            )
            mispriced_contracts.append(mispriced_contract)
        else:
            mispriced_contracts.append(contract)
    
    return mispriced_contracts

def create_sample_portfolio_scenario(scenario: str = "balanced") -> tuple[List[OptionContract], str]:
    """
    Create predefined scenarios for testing the optimizer.
    
    Args:
        scenario: One of "balanced", "high_vol", "low_vol", "skewed"
    
    Returns:
        Tuple of (contracts_list, scenario_description)
    """
    if scenario == "balanced":
        # Balanced market with moderate volatility
        contracts = generate_mock_options_chain(
            underlying_price=450.0,
            base_implied_vol=0.20,
            vol_skew=0.02,
            expiration_days=[30, 60, 90]
        )
        description = "Balanced market scenario with 20% base volatility"
        
    elif scenario == "high_vol":
        # High volatility environment
        contracts = generate_mock_options_chain(
            underlying_price=450.0,
            base_implied_vol=0.35,
            vol_skew=0.05,
            expiration_days=[15, 30, 45]
        )
        description = "High volatility scenario with 35% base volatility"
        
    elif scenario == "low_vol":
        # Low volatility environment
        contracts = generate_mock_options_chain(
            underlying_price=450.0,
            base_implied_vol=0.12,
            vol_skew=0.01,
            expiration_days=[30, 60, 90, 120]
        )
        description = "Low volatility scenario with 12% base volatility"
        
    elif scenario == "skewed":
        # Market with strong volatility skew
        contracts = generate_mock_options_chain(
            underlying_price=450.0,
            base_implied_vol=0.18,
            vol_skew=0.08,  # Strong skew
            expiration_days=[30, 60]
        )
        description = "Skewed market with strong put volatility premium"
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Add some mispricing to create opportunities
    contracts = generate_options_with_mispricing(contracts)
    
    return contracts, description

def contracts_to_dataframe(contracts: List[OptionContract]) -> pd.DataFrame:
    """Convert list of option contracts to DataFrame for analysis."""
    data = []
    for contract in contracts:
        data.append({
            'Symbol': contract.symbol,
            'Type': contract.option_type,
            'Strike': contract.strike,
            'Expiration': contract.expiration,
            'Days to Expiry': (contract.expiration - date.today()).days,
            'Underlying Price': contract.underlying_price,
            'Market Premium': contract.market_premium,
            'Implied Vol': contract.implied_volatility,
            'Delta': contract.delta,
            'Gamma': contract.gamma,
            'Vega': contract.vega,
            'Theta': contract.theta,
            'Moneyness': contract.underlying_price / contract.strike,
            'ITM': ((contract.option_type == 'call' and contract.underlying_price > contract.strike) or
                   (contract.option_type == 'put' and contract.underlying_price < contract.strike))
        })
    
    df = pd.DataFrame(data)
    return df

def filter_contracts_for_optimization(
    contracts: List[OptionContract],
    max_days_to_expiry: int = 120,
    min_days_to_expiry: int = 7,
    min_premium: float = 0.50,
    strike_range_pct: float = 0.20  # ±20% from underlying price
) -> List[OptionContract]:
    """
    Filter contracts to a reasonable set for optimization.
    
    This removes very short-term, very long-term, very cheap, or very far OTM options
    that might not be suitable for the optimization strategy.
    """
    filtered = []
    current_date = date.today()
    
    for contract in contracts:
        days_to_expiry = (contract.expiration - current_date).days
        
        # Filter by time to expiry
        if days_to_expiry < min_days_to_expiry or days_to_expiry > max_days_to_expiry:
            continue
        
        # Filter by premium
        if contract.market_premium < min_premium:
            continue
        
        # Filter by strike range (keep strikes within ±20% of underlying)
        strike_range = contract.underlying_price * strike_range_pct
        if (contract.strike < contract.underlying_price - strike_range or 
            contract.strike > contract.underlying_price + strike_range):
            continue
        
        filtered.append(contract)
    
    return filtered

# Predefined test data for quick testing
def get_small_test_chain() -> List[OptionContract]:
    """Get a small options chain for quick testing (10-15 contracts)."""
    contracts = generate_mock_options_chain(
        underlying_price=100.0,
        strike_range=(90, 110),
        strike_increment=5.0,
        expiration_days=[30],
        base_implied_vol=0.20
    )
    return filter_contracts_for_optimization(contracts, max_days_to_expiry=45) 