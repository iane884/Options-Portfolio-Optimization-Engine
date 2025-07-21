import numpy as np
from scipy.stats import norm
from typing import List
from models import Portfolio, OptionContract, Position

def calculate_portfolio_pnl_stats(
    portfolio: Portfolio,
    underlying_volatility: float = 0.20,
    time_horizon_days: int = 1
) -> tuple[float, float]:
    """
    Calculate portfolio P&L mean and standard deviation using delta-gamma-normal approximation.
    
    For simplicity, we approximate portfolio P&L standard deviation using:
    1. Delta exposure to underlying moves
    2. Individual position risk contribution
    
    Args:
        portfolio: Portfolio to analyze
        underlying_volatility: Annualized volatility of underlying asset
        time_horizon_days: Time horizon for risk calculation in days
    
    Returns:
        Tuple of (mean_pnl, std_pnl)
    """
    if not portfolio.positions:
        return 0.0, 0.0
    
    # Expected P&L is the sum of expected returns from all positions
    mean_pnl = portfolio.expected_return
    
    # Calculate portfolio standard deviation
    # For delta-gamma approximation, dominant risk comes from:
    # 1. Delta exposure to underlying price moves
    # 2. Individual position volatilities (simplified)
    
    # Get portfolio delta
    portfolio_delta = portfolio.total_delta
    
    # Assume underlying price for risk calculation
    underlying_price = portfolio.positions[0].contract.underlying_price  # Assume all same underlying
    
    # Daily volatility
    daily_vol = underlying_volatility / np.sqrt(252)  # Convert annualized to daily
    
    # Risk from delta exposure (linear approximation)
    # Delta risk = |portfolio_delta| * underlying_price * daily_volatility
    delta_risk = abs(portfolio_delta) * underlying_price * daily_vol
    
    # Risk from individual positions (simplified approach)
    # For each position, estimate daily P&L volatility
    position_risks = []
    for position in portfolio.positions:
        # Risk per contract based on vega and underlying moves
        # Simplified: use vega as a proxy for volatility sensitivity
        contract_risk = abs(position.contract.vega) * underlying_volatility / 100  # Vega is per 1% vol change
        position_risk = abs(position.quantity) * contract_risk
        position_risks.append(position_risk)
    
    # Total position risk (assuming uncorrelated for simplicity)
    total_position_risk = np.sqrt(sum(risk**2 for risk in position_risks))
    
    # Combine delta risk and position risk
    # In reality, these are correlated, but for MVP we'll add them in quadrature
    total_risk = np.sqrt(delta_risk**2 + total_position_risk**2)
    
    # Scale by time horizon if different from 1 day
    if time_horizon_days != 1:
        total_risk *= np.sqrt(time_horizon_days)
    
    return mean_pnl, total_risk

def calculate_cvar_normal(
    mean: float,
    std: float,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) for a normal distribution.
    
    CVaR is the expected loss in the worst (1-confidence_level) tail of the distribution.
    
    Args:
        mean: Mean of P&L distribution
        std: Standard deviation of P&L distribution  
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        CVaR value (negative means loss)
    """
    if std <= 0:
        return mean
    
    # Calculate the quantile for the tail
    alpha = 1 - confidence_level  # e.g., 0.05 for 95% confidence
    z_alpha = norm.ppf(alpha)  # Standard normal quantile
    
    # Calculate CVaR using the closed-form formula for normal distribution
    phi_z = norm.pdf(z_alpha)  # Standard normal density
    cvar = mean - std * phi_z / alpha
    
    return cvar

def calculate_var_normal(
    mean: float,
    std: float,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) for a normal distribution.
    
    VaR is the loss threshold that will not be exceeded with the given confidence level.
    
    Returns:
        VaR value (negative means loss)
    """
    if std <= 0:
        return mean
    
    alpha = 1 - confidence_level
    z_alpha = norm.ppf(alpha)
    var = mean + std * z_alpha  # Note: z_alpha is negative for typical confidence levels
    
    return var

def calculate_portfolio_risk_metrics(
    portfolio: Portfolio,
    underlying_volatility: float = 0.20,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1
) -> dict:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Returns:
        Dictionary with risk metrics including CVaR, VaR, volatility, etc.
    """
    # Calculate P&L statistics
    mean_pnl, std_pnl = calculate_portfolio_pnl_stats(
        portfolio, underlying_volatility, time_horizon_days
    )
    
    # Calculate CVaR and VaR
    cvar = calculate_cvar_normal(mean_pnl, std_pnl, confidence_level)
    var = calculate_var_normal(mean_pnl, std_pnl, confidence_level)
    
    # Calculate additional metrics
    sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
    
    return {
        'expected_pnl': mean_pnl,
        'pnl_volatility': std_pnl,
        'cvar': cvar,
        'var': var,
        'sharpe_ratio': sharpe_ratio,
        'confidence_level': confidence_level,
        'time_horizon_days': time_horizon_days,
        'portfolio_delta': portfolio.total_delta,
        'portfolio_gamma': portfolio.total_gamma,
        'portfolio_vega': portfolio.total_vega
    }

def check_cvar_constraint(
    portfolio: Portfolio,
    cvar_limit: float,
    underlying_volatility: float = 0.20,
    confidence_level: float = 0.95
) -> tuple[bool, float]:
    """
    Check if portfolio satisfies CVaR constraint.
    
    Args:
        portfolio: Portfolio to check
        cvar_limit: Maximum acceptable CVaR (typically negative)
        underlying_volatility: Underlying asset volatility
        confidence_level: Confidence level for CVaR calculation
    
    Returns:
        Tuple of (constraint_satisfied, actual_cvar)
    """
    risk_metrics = calculate_portfolio_risk_metrics(
        portfolio, underlying_volatility, confidence_level
    )
    
    actual_cvar = risk_metrics['cvar']
    constraint_satisfied = actual_cvar >= cvar_limit
    
    return constraint_satisfied, actual_cvar

def estimate_portfolio_volatility_simple(
    positions: List[Position],
    underlying_volatility: float = 0.20
) -> float:
    """
    Simple portfolio volatility estimation for optimization constraints.
    
    This is a simplified version that can be used in linear constraints
    by avoiding the square root in the full variance calculation.
    
    Returns sum of individual position risk contributions.
    """
    total_risk = 0.0
    
    for position in positions:
        # Risk contribution based on delta and vega exposure
        delta_risk = abs(position.quantity * position.contract.delta * 
                        position.contract.underlying_price * underlying_volatility / np.sqrt(252))
        
        vega_risk = abs(position.quantity * position.contract.vega * underlying_volatility / 100)
        
        # Add position risk (simplified linear combination)
        position_risk = delta_risk + vega_risk
        total_risk += position_risk
    
    return total_risk

def monte_carlo_portfolio_pnl(
    portfolio: Portfolio,
    underlying_volatility: float = 0.20,
    num_simulations: int = 10000,
    time_horizon_days: int = 1
) -> np.ndarray:
    """
    Monte Carlo simulation of portfolio P&L (alternative to normal approximation).
    
    This function is included for completeness but not used in the MVP optimization.
    It provides a more accurate risk assessment that could replace the normal approximation.
    
    Returns:
        Array of simulated P&L values
    """
    if not portfolio.positions:
        return np.array([0.0])
    
    # Simulate underlying price movements
    underlying_price = portfolio.positions[0].contract.underlying_price
    daily_vol = underlying_volatility / np.sqrt(252)
    
    # Generate random returns
    random_returns = np.random.normal(0, daily_vol * np.sqrt(time_horizon_days), num_simulations)
    simulated_prices = underlying_price * np.exp(random_returns)
    
    # Calculate P&L for each simulation
    pnl_simulations = []
    
    for sim_price in simulated_prices:
        portfolio_pnl = 0.0
        
        for position in portfolio.positions:
            # Simplified P&L calculation using delta approximation
            price_change = sim_price - underlying_price
            option_pnl = position.quantity * position.contract.delta * price_change * position.contract.contract_multiplier
            
            # Add expected return component
            option_pnl += position.expected_pnl
            
            portfolio_pnl += option_pnl
        
        pnl_simulations.append(portfolio_pnl)
    
    return np.array(pnl_simulations)

def calculate_cvar_monte_carlo(
    pnl_simulations: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate CVaR from Monte Carlo simulation results.
    
    Returns:
        CVaR value (expected loss in worst tail)
    """
    if len(pnl_simulations) == 0:
        return 0.0
    
    # Sort P&L values
    sorted_pnl = np.sort(pnl_simulations)
    
    # Find the tail threshold
    tail_size = int((1 - confidence_level) * len(sorted_pnl))
    if tail_size == 0:
        tail_size = 1
    
    # Calculate CVaR as average of worst tail
    cvar = np.mean(sorted_pnl[:tail_size])
    
    return cvar 