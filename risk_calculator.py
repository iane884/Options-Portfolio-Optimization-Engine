import numpy as np
from scipy.stats import norm
from typing import List, Tuple
from models import Portfolio, OptionContract, Position

def calculate_portfolio_pnl_stats(
    portfolio: Portfolio,
    underlying_volatility: float = 0.20,
    time_horizon_days: int = 1
) -> Tuple[float, float]:
    """
    Calculate portfolio P&L mean and standard deviation using a
    delta-gamma-vega approximation.

    Improvements over the previous version
    --------------------------------------
    1.  Includes gamma-risk (second-order price moves).
    2.  Uses contract_multiplier so dollar risks are correctly scaled.
    3.  Enforces a small volatility floor to avoid division by ~0 which
        produced crazy Sharpe ratios.
    """
    if not portfolio.positions:
        return 0.0, 0.0

    mean_pnl = portfolio.expected_return

    # Annual->daily σ of the underlying
    daily_vol = underlying_volatility / np.sqrt(252)

    risk_sq_sum = 0.0
    for pos in portfolio.positions:
        q            = abs(pos.quantity)
        contract     = pos.contract
        mult         = contract.contract_multiplier
        spot         = contract.underlying_price

        # 1) Delta risk  (linear P&L from move dS)
        delta_risk   = q * abs(contract.delta)  * spot * daily_vol

        # 2) Gamma risk (convexity P&L from move dS)
        gamma_risk   = q * 0.5 * abs(contract.gamma) * (spot * daily_vol)**2

        # 3) Vega risk  (vol-change of 1 vol-point = 1 %)
        # Note: contract.vega is already per 1% vol change, so use vol change of ~5% for risk
        vega_risk    = q * abs(contract.vega) * 0.05  # 5% volatility change for risk

        # Combine for *that* position (uncorrelated assumption)
        position_risk = np.sqrt(delta_risk**2 + gamma_risk**2 + vega_risk**2) * mult
        risk_sq_sum  += position_risk**2

    std_pnl = np.sqrt(risk_sq_sum)

    # Volatility floor – prevents divide-by-(almost)-zero
    min_vol = 100.0          # $100 per day ≈ 0.02% of a $500 k book
    std_pnl = max(std_pnl, min_vol)

    # Scale for different horizons
    if time_horizon_days != 1:
        std_pnl *= np.sqrt(time_horizon_days)

    return mean_pnl, std_pnl

def calculate_cvar_normal(
    mean: float,
    std: float,
    confidence_level: float = 0.95
) -> float:
    """
    CVaR (expected loss in the worst (1-α) tail) for a normal
    distribution.  Returns a *negative* number for loss.
    """
    if std <= 0:
        return 0.0

    alpha   = 1 - confidence_level          # e.g. 0.05
    z_alpha = norm.ppf(confidence_level)    # +1.645 for 95 %
    phi_z   = norm.pdf(z_alpha)

    var  = mean - std * z_alpha             # 95 % VaR
    cvar = mean - std * phi_z / alpha       # 95 % CVaR

    # We are interested in loss → negative value (or 0 if profitable)
    return min(cvar, 0.0)

def calculate_var_normal(
    mean: float,
    std: float,
    confidence_level: float = 0.95
) -> float:
    """Same sign convention as CVaR (negative = loss)."""
    if std <= 0:
        return 0.0

    z_alpha = norm.ppf(confidence_level)    # +1.645
    var     = mean - std * z_alpha
    return min(var, 0.0)

def calculate_portfolio_risk_metrics(
    portfolio: Portfolio,
    underlying_volatility: float = 0.20,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1
) -> dict:
    # --- P&L statistics --------------------------------------------------
    mean_pnl, std_pnl = calculate_portfolio_pnl_stats(
        portfolio, underlying_volatility, time_horizon_days
    )

    # --- Tail-risk -------------------------------------------------------
    cvar = calculate_cvar_normal(mean_pnl, std_pnl, confidence_level)
    var  = calculate_var_normal (mean_pnl, std_pnl, confidence_level)

    # --- Sharpe ratio ----------------------------------------------------
    sharpe_ratio = mean_pnl / std_pnl
    sharpe_ratio = min(sharpe_ratio, 5.0)  # cap at a realistic level

    return {
        'expected_pnl'     : mean_pnl,
        'pnl_volatility'   : std_pnl,
        'cvar'             : cvar,
        'var'              : var,
        'sharpe_ratio'     : sharpe_ratio,
        'confidence_level' : confidence_level,
        'time_horizon_days': time_horizon_days,
        'portfolio_delta'  : portfolio.total_delta,
        'portfolio_gamma'  : portfolio.total_gamma,
        'portfolio_vega'   : portfolio.total_vega,
    }

def check_cvar_constraint(
    portfolio: Portfolio,
    cvar_limit: float,
    underlying_volatility: float = 0.20,
    confidence_level: float = 0.95
) -> Tuple[bool, float]:
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