import numpy as np
from scipy.stats import norm
from datetime import date, datetime
from typing import List
from models import OptionContract

def black_scholes_price(
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        underlying_price: Current price of underlying asset
        strike: Strike price of option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Option price according to Black-Scholes formula
    """
    if time_to_expiry <= 0:
        # Option has expired, return intrinsic value
        if option_type == 'call':
            return max(0, underlying_price - strike)
        else:  # put
            return max(0, strike - underlying_price)
    
    d1 = (np.log(underlying_price / strike) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    if option_type == 'call':
        price = (underlying_price * norm.cdf(d1) - 
                strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
    else:  # put
        price = (strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                underlying_price * norm.cdf(-d1))
    
    return max(0, price)

def calculate_greeks(
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str
) -> dict:
    """
    Calculate option Greeks using Black-Scholes formulas.
    
    Returns:
        Dictionary with delta, gamma, vega, theta
    """
    if time_to_expiry <= 0:
        # Option has expired
        if option_type == 'call':
            delta = 1.0 if underlying_price > strike else 0.0
        else:  # put
            delta = -1.0 if underlying_price < strike else 0.0
        return {'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
    
    d1 = (np.log(underlying_price / strike) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1.0
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (underlying_price * volatility * np.sqrt(time_to_expiry))
    
    # Vega (same for calls and puts)
    vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100  # Per 1% volatility change
    
    # Theta
    common_term = (-underlying_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)))
    if option_type == 'call':
        theta = (common_term - 
                risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
    else:  # put
        theta = (common_term + 
                risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }

def time_to_expiry_years(expiration_date: date, current_date: date = None) -> float:
    """Calculate time to expiry in years."""
    if current_date is None:
        current_date = date.today()
    
    days_to_expiry = (expiration_date - current_date).days
    return max(0, days_to_expiry / 365.25)

def calculate_expected_returns(
    contracts: List[OptionContract],
    risk_free_rate: float = 0.05,
    volatility_override: float = None
) -> List[OptionContract]:
    """
    Calculate expected returns for option contracts using Black-Scholes theoretical pricing.
    
    Expected return = Market Premium - Theoretical Price
    
    Args:
        contracts: List of option contracts
        risk_free_rate: Risk-free rate for Black-Scholes calculation
        volatility_override: If provided, use this volatility instead of implied vol for theoretical pricing
    
    Returns:
        List of contracts with theoretical_price and expected_return populated
    """
    updated_contracts = []
    
    for contract in contracts:
        # Calculate time to expiry
        time_to_expiry = time_to_expiry_years(contract.expiration)
        
        # Use volatility override if provided, otherwise use implied volatility
        vol_for_pricing = volatility_override if volatility_override is not None else contract.implied_volatility
        
        # Calculate theoretical price using Black-Scholes
        theoretical_price = black_scholes_price(
            underlying_price=contract.underlying_price,
            strike=contract.strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=vol_for_pricing,
            option_type=contract.option_type
        )
        
        # Calculate expected return (positive means option is overpriced, good to sell)
        expected_return = contract.market_premium - theoretical_price
        
        # Create updated contract
        updated_contract = OptionContract(
            symbol=contract.symbol,
            strike=contract.strike,
            expiration=contract.expiration,
            option_type=contract.option_type,
            underlying_price=contract.underlying_price,
            market_premium=contract.market_premium,
            implied_volatility=contract.implied_volatility,
            delta=contract.delta,
            gamma=contract.gamma,
            vega=contract.vega,
            theta=contract.theta,
            contract_multiplier=contract.contract_multiplier,
            theoretical_price=theoretical_price,
            expected_return=expected_return
        )
        
        updated_contracts.append(updated_contract)
    
    return updated_contracts

def update_greeks_for_contracts(
    contracts: List[OptionContract],
    risk_free_rate: float = 0.05
) -> List[OptionContract]:
    """
    Recalculate Greeks for option contracts using Black-Scholes formulas.
    
    This is useful if the input contracts have inaccurate or missing Greeks.
    """
    updated_contracts = []
    
    for contract in contracts:
        time_to_expiry = time_to_expiry_years(contract.expiration)
        
        greeks = calculate_greeks(
            underlying_price=contract.underlying_price,
            strike=contract.strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=contract.implied_volatility,
            option_type=contract.option_type
        )
        
        # Create updated contract with recalculated Greeks
        updated_contract = OptionContract(
            symbol=contract.symbol,
            strike=contract.strike,
            expiration=contract.expiration,
            option_type=contract.option_type,
            underlying_price=contract.underlying_price,
            market_premium=contract.market_premium,
            implied_volatility=contract.implied_volatility,
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            vega=greeks['vega'],
            theta=greeks['theta'],
            contract_multiplier=contract.contract_multiplier,
            theoretical_price=contract.theoretical_price,
            expected_return=contract.expected_return
        )
        
        updated_contracts.append(updated_contract)
    
    return updated_contracts 