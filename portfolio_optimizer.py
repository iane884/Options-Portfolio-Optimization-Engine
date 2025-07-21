from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import date

from models import (
    OptionContract, Portfolio, OptimizationConstraints, OptimizationResult
)
from data_generator import (
    create_sample_portfolio_scenario, filter_contracts_for_optimization,
    contracts_to_dataframe
)
from pricing import calculate_expected_returns, update_greeks_for_contracts
from risk_calculator import calculate_portfolio_risk_metrics
from optimizers import MILPOptimizer, GreedyOptimizer, compare_optimizers

class PortfolioOptimizer:
    """
    Main orchestrator class for options portfolio optimization.
    
    This class provides a high-level interface for:
    1. Loading and preprocessing option contract data
    2. Setting optimization constraints
    3. Running optimization algorithms
    4. Analyzing and reporting results
    """
    
    def __init__(self, constraints: OptimizationConstraints = None):
        """
        Initialize the portfolio optimizer.
        
        Args:
            constraints: Optimization constraints. If None, uses default values.
        """
        self.constraints = constraints or OptimizationConstraints()
        self.contracts: List[OptionContract] = []
        self.underlying_volatility = 0.20
        self.last_result: Optional[OptimizationResult] = None
        
    def load_contracts_from_scenario(self, scenario: str = "balanced") -> Tuple[int, str]:
        """
        Load option contracts from a predefined scenario.
        
        Args:
            scenario: One of "balanced", "high_vol", "low_vol", "skewed"
        
        Returns:
            Tuple of (number_of_contracts, scenario_description)
        """
        contracts, description = create_sample_portfolio_scenario(scenario)
        
        # Filter to reasonable set for optimization
        filtered_contracts = filter_contracts_for_optimization(contracts)
        
        # Calculate expected returns
        self.contracts = calculate_expected_returns(filtered_contracts)
        
        return len(self.contracts), description
    
    def load_contracts(self, contracts: List[OptionContract]) -> int:
        """
        Load option contracts directly.
        
        Args:
            contracts: List of option contracts
        
        Returns:
            Number of contracts loaded
        """
        # Ensure Greeks are calculated
        contracts = update_greeks_for_contracts(contracts)
        
        # Calculate expected returns
        self.contracts = calculate_expected_returns(contracts)
        
        return len(self.contracts)
    
    def set_constraints(
        self,
        delta_tolerance: float = None,
        gamma_tolerance: float = None,
        capital_budget: float = None,
        margin_cap: float = None,
        cvar_limit: float = None,
        **kwargs
    ):
        """
        Update optimization constraints.
        
        Args:
            delta_tolerance: Maximum absolute portfolio delta
            gamma_tolerance: Maximum absolute portfolio gamma
            capital_budget: Maximum capital to deploy
            margin_cap: Maximum margin usage
            cvar_limit: Maximum acceptable CVaR (negative value)
            **kwargs: Other constraint parameters
        """
        if delta_tolerance is not None:
            self.constraints.delta_tolerance = delta_tolerance
        if gamma_tolerance is not None:
            self.constraints.gamma_tolerance = gamma_tolerance
        if capital_budget is not None:
            self.constraints.capital_budget = capital_budget
        if margin_cap is not None:
            self.constraints.margin_cap = margin_cap
        if cvar_limit is not None:
            self.constraints.cvar_limit = cvar_limit
            
        # Update other parameters
        for key, value in kwargs.items():
            if hasattr(self.constraints, key):
                setattr(self.constraints, key, value)
    
    def set_underlying_volatility(self, volatility: float):
        """Set the underlying asset volatility for risk calculations."""
        self.underlying_volatility = volatility
    
    def optimize(
        self,
        method: str = "milp",
        allow_fractional: bool = False,
        **kwargs
    ) -> OptimizationResult:
        """
        Run portfolio optimization.
        
        Args:
            method: Optimization method ("milp", "greedy", or "both")
            allow_fractional: Allow fractional contract quantities (MILP only)
            **kwargs: Additional optimizer parameters
        
        Returns:
            OptimizationResult with optimal portfolio and metrics
        """
        if not self.contracts:
            raise ValueError("No contracts loaded. Use load_contracts() or load_contracts_from_scenario() first.")
        
        if method == "milp":
            optimizer = MILPOptimizer(self.constraints)
            result = optimizer.optimize(
                self.contracts, 
                self.underlying_volatility, 
                allow_fractional=allow_fractional,
                **kwargs
            )
        elif method == "greedy":
            optimizer = GreedyOptimizer(self.constraints)
            result = optimizer.optimize(
                self.contracts, 
                self.underlying_volatility,
                **kwargs
            )
        elif method == "both":
            results = compare_optimizers(
                self.contracts, 
                self.constraints, 
                self.underlying_volatility
            )
            # Return the better result (higher objective value)
            if results['milp'].objective_value >= results['greedy'].objective_value:
                result = results['milp']
            else:
                result = results['greedy']
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        self.last_result = result
        return result
    
    def get_contracts_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of available contracts."""
        if not self.contracts:
            return pd.DataFrame()
        
        return contracts_to_dataframe(self.contracts)
    
    def get_portfolio_summary(self, result: OptimizationResult = None) -> Dict:
        """
        Get a comprehensive summary of the optimized portfolio.
        
        Args:
            result: OptimizationResult to analyze. If None, uses last result.
        
        Returns:
            Dictionary with portfolio summary metrics
        """
        if result is None:
            result = self.last_result
        
        if result is None or not result.portfolio.positions:
            return {"error": "No optimization result available"}
        
        portfolio = result.portfolio
        
        # Calculate comprehensive risk metrics
        risk_metrics = calculate_portfolio_risk_metrics(
            portfolio, 
            self.underlying_volatility, 
            self.constraints.cvar_confidence
        )
        
        # Portfolio composition
        composition = []
        for pos in portfolio.positions:
            composition.append({
                'Symbol': pos.contract.symbol,
                'Type': pos.contract.option_type,
                'Strike': pos.contract.strike,
                'Expiration': pos.contract.expiration,
                'Quantity': pos.quantity,
                'Position': 'Long' if pos.is_long else 'Short',
                'Premium Cost': pos.premium_cost,
                'Expected P&L': pos.expected_pnl,
                'Delta Contribution': pos.quantity * pos.contract.delta,
                'Gamma Contribution': pos.quantity * pos.contract.gamma
            })
        
        # Constraint satisfaction
        constraints_status = {
            'Delta': {
                'Value': f"{portfolio.total_delta:.3f}",
                'Limit': f"±{self.constraints.delta_tolerance}",
                'Satisfied': abs(portfolio.total_delta) <= self.constraints.delta_tolerance
            },
            'Gamma': {
                'Value': f"{portfolio.total_gamma:.3f}",
                'Limit': f"±{self.constraints.gamma_tolerance}",
                'Satisfied': abs(portfolio.total_gamma) <= self.constraints.gamma_tolerance
            },
            'Capital Used': {
                'Value': f"${portfolio.capital_used:,.0f}",
                'Limit': f"${self.constraints.capital_budget:,.0f}",
                'Satisfied': portfolio.capital_used <= self.constraints.capital_budget
            },
            'Margin Used': {
                'Value': f"${portfolio.margin_used(self.constraints.margin_per_contract):,.0f}",
                'Limit': f"${self.constraints.margin_cap:,.0f}",
                'Satisfied': portfolio.margin_used(self.constraints.margin_per_contract) <= self.constraints.margin_cap
            },
            'CVaR': {
                'Value': f"${risk_metrics['cvar']:,.0f}",
                'Limit': f"${self.constraints.cvar_limit:,.0f}",
                'Satisfied': risk_metrics['cvar'] >= self.constraints.cvar_limit
            }
        }
        
        return {
            'optimization_status': result.optimization_status,
            'objective_value': result.objective_value,
            'constraints_satisfied': result.constraints_satisfied,
            'portfolio_metrics': {
                'Total Positions': len(portfolio.positions),
                'Net Premium': portfolio.net_premium,
                'Expected Return': portfolio.expected_return,
                'Portfolio Delta': portfolio.total_delta,
                'Portfolio Gamma': portfolio.total_gamma,
                'Portfolio Vega': portfolio.total_vega,
                'Capital Used': portfolio.capital_used,
                'Margin Used': portfolio.margin_used(self.constraints.margin_per_contract)
            },
            'risk_metrics': risk_metrics,
            'constraints_status': constraints_status,
            'portfolio_composition': composition
        }
    
    def analyze_scenario_sensitivity(
        self, 
        underlying_price_changes: List[float] = None,
        volatility_changes: List[float] = None
    ) -> Dict:
        """
        Analyze how the portfolio performs under different market scenarios.
        
        Args:
            underlying_price_changes: List of percentage changes in underlying price
            volatility_changes: List of percentage changes in volatility
        
        Returns:
            Dictionary with scenario analysis results
        """
        if self.last_result is None:
            return {"error": "No optimization result available"}
        
        if underlying_price_changes is None:
            underlying_price_changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        
        if volatility_changes is None:
            volatility_changes = [-0.25, 0.0, 0.25]
        
        portfolio = self.last_result.portfolio
        if not portfolio.positions:
            return {"error": "Empty portfolio"}
        
        # Get current underlying price
        current_price = portfolio.positions[0].contract.underlying_price
        
        scenarios = []
        
        for price_change in underlying_price_changes:
            new_price = current_price * (1 + price_change)
            
            for vol_change in volatility_changes:
                new_vol = self.underlying_volatility * (1 + vol_change)
                
                # Calculate approximate P&L using delta approximation
                price_move = new_price - current_price
                delta_pnl = portfolio.total_delta * price_move * 100  # 100 shares per contract
                
                # Add expected return
                total_pnl = delta_pnl + portfolio.expected_return
                
                scenarios.append({
                    'Price Change %': price_change * 100,
                    'New Price': new_price,
                    'Vol Change %': vol_change * 100,
                    'New Volatility': new_vol,
                    'Estimated P&L': total_pnl,
                    'Delta P&L': delta_pnl,
                    'Expected Return': portfolio.expected_return
                })
        
        return {
            'scenarios': scenarios,
            'current_price': current_price,
            'current_volatility': self.underlying_volatility,
            'portfolio_delta': portfolio.total_delta,
            'portfolio_gamma': portfolio.total_gamma
        }
    
    def export_portfolio_to_csv(self, filename: str = None) -> str:
        """
        Export the optimized portfolio to a CSV file.
        
        Args:
            filename: Output filename. If None, generates default name.
        
        Returns:
            Filename where portfolio was saved
        """
        if self.last_result is None or not self.last_result.portfolio.positions:
            raise ValueError("No portfolio to export")
        
        if filename is None:
            filename = f"optimized_portfolio_{date.today().strftime('%Y%m%d')}.csv"
        
        # Create DataFrame from portfolio
        df = self.last_result.portfolio.to_dataframe()
        
        # Add summary information at the top
        summary_rows = [
            ['PORTFOLIO SUMMARY', '', '', '', '', '', '', '', '', '', ''],
            ['Optimization Status', self.last_result.optimization_status, '', '', '', '', '', '', '', '', ''],
            ['Objective Value', self.last_result.objective_value, '', '', '', '', '', '', '', '', ''],
            ['Constraints Satisfied', self.last_result.constraints_satisfied, '', '', '', '', '', '', '', '', ''],
            ['Portfolio Delta', self.last_result.portfolio.total_delta, '', '', '', '', '', '', '', '', ''],
            ['Portfolio Gamma', self.last_result.portfolio.total_gamma, '', '', '', '', '', '', '', '', ''],
            ['Expected Return', self.last_result.portfolio.expected_return, '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', '', '', ''],  # Empty row
            ['POSITIONS', '', '', '', '', '', '', '', '', '', '']
        ]
        
        # Combine summary and portfolio data
        summary_df = pd.DataFrame(summary_rows, columns=df.columns)
        full_df = pd.concat([summary_df, df], ignore_index=True)
        
        # Save to CSV
        full_df.to_csv(filename, index=False)
        
        return filename 