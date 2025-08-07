import numpy as np
import cvxpy as cp
from typing import List, Dict, Optional, Tuple
from models import OptionContract, Portfolio, Position, OptimizationConstraints, OptimizationResult
from pricing import calculate_expected_returns
from risk_calculator import calculate_portfolio_risk_metrics, check_cvar_constraint

class MILPOptimizer:
    """Mixed Integer Linear Programming optimizer using CVXPY."""
    
    def __init__(self, constraints: OptimizationConstraints):
        self.constraints = constraints
    
    def optimize(
        self,
        contracts: List[OptionContract],
        underlying_volatility: float = 0.20,
        allow_fractional: bool = False
    ) -> OptimizationResult:
        """
        Solve the portfolio optimization problem using MILP.
        
        Args:
            contracts: List of available option contracts
            underlying_volatility: Underlying asset volatility for risk calculation
            allow_fractional: If True, allow fractional contract quantities
        
        Returns:
            OptimizationResult with optimal portfolio and metrics
        """
        n_contracts = len(contracts)
        
        if n_contracts == 0:
            return self._create_empty_result("No contracts provided")
        
        # Ensure all contracts have expected returns calculated
        if any(c.expected_return is None for c in contracts):
            contracts = calculate_expected_returns(contracts)
        
        # Decision variables: quantity of each contract to hold
        if allow_fractional:
            q = cp.Variable(n_contracts, name="quantities")
        else:
            q = cp.Variable(n_contracts, integer=True, name="quantities")
        
        # Extract data arrays
        expected_returns = np.array([c.expected_return for c in contracts])
        deltas = np.array([c.delta for c in contracts])
        gammas = np.array([c.gamma for c in contracts])
        premiums = np.array([c.market_premium * c.contract_multiplier for c in contracts])
        
        # Objective: maximize expected return (already includes contract multiplier in expected_returns)
        objective = cp.Maximize(expected_returns @ q)
        
        # Constraints
        constraints = []
        
        # Position size bounds
        constraints += [
            q >= self.constraints.min_contracts_per_option,
            q <= self.constraints.max_contracts_per_option
        ]
        
        # Delta neutrality constraint
        constraints += [
            deltas @ q >= -self.constraints.delta_tolerance,
            deltas @ q <= self.constraints.delta_tolerance
        ]
        
        # Gamma neutrality constraint
        constraints += [
            gammas @ q >= -self.constraints.gamma_tolerance,
            gammas @ q <= self.constraints.gamma_tolerance
        ]
        
        # Capital constraint (net premium paid for long positions)
        long_costs = cp.sum(cp.pos(q) * premiums)
        constraints += [long_costs <= self.constraints.capital_budget]
        
        # Margin constraint (based on short positions)
        short_contracts = cp.sum(cp.neg(q))  # Number of short contracts
        margin_used = short_contracts * self.constraints.margin_per_contract
        constraints += [margin_used <= self.constraints.margin_cap]
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        try:
            # Try different solvers in order of preference
            solvers_to_try = [cp.CPLEX, cp.GUROBI, cp.CBC, cp.GLPK_MI] if not allow_fractional else [cp.CPLEX, cp.GUROBI, cp.CLARABEL, cp.OSQP]
            
            solved = False
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver, verbose=False)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solved = True
                        break
                except:
                    continue
            
            if not solved:
                # Fallback to default solver
                problem.solve(verbose=False)
            
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self._create_empty_result(f"Optimization failed: {problem.status}")
            
            # Extract solution
            quantities = q.value
            if quantities is None:
                return self._create_empty_result("No solution found")
            
            # Round if integer solution was requested but we got fractional
            if not allow_fractional:
                quantities = np.round(quantities).astype(int)
            
            # Create portfolio from solution
            portfolio = self._create_portfolio_from_solution(contracts, quantities)
            
            # Calculate risk metrics
            risk_metrics = calculate_portfolio_risk_metrics(
                portfolio, underlying_volatility, self.constraints.cvar_confidence
            )
            
            # Check constraints
            constraints_satisfied, violations = self._check_constraints(portfolio, risk_metrics)
            
            return OptimizationResult(
                portfolio=portfolio,
                objective_value=problem.value or 0.0,
                optimization_status=problem.status,
                constraints_satisfied=constraints_satisfied,
                portfolio_cvar=risk_metrics['cvar'],
                portfolio_std=risk_metrics['pnl_volatility'],
                **violations
            )
            
        except Exception as e:
            return self._create_empty_result(f"Optimization error: {str(e)}")
    
    def _create_portfolio_from_solution(
        self, 
        contracts: List[OptionContract], 
        quantities: np.ndarray
    ) -> Portfolio:
        """Create Portfolio object from optimization solution."""
        positions = []
        for i, quantity in enumerate(quantities):
            if abs(quantity) >= 0.1:  # Only include non-zero positions
                positions.append(Position(contract=contracts[i], quantity=int(quantity)))
        
        return Portfolio(positions=positions)
    
    def _check_constraints(self, portfolio: Portfolio, risk_metrics: dict) -> Tuple[bool, dict]:
        """Check if solution satisfies all constraints."""
        violations = {
            'delta_violation': 0.0,
            'gamma_violation': 0.0,
            'capital_violation': 0.0,
            'margin_violation': 0.0,
            'cvar_violation': 0.0
        }
        
        # Check delta constraint
        delta = portfolio.total_delta
        if abs(delta) > self.constraints.delta_tolerance:
            violations['delta_violation'] = abs(delta) - self.constraints.delta_tolerance
        
        # Check gamma constraint
        gamma = portfolio.total_gamma
        if abs(gamma) > self.constraints.gamma_tolerance:
            violations['gamma_violation'] = abs(gamma) - self.constraints.gamma_tolerance
        
        # Check capital constraint
        capital_used = portfolio.capital_used
        if capital_used > self.constraints.capital_budget:
            violations['capital_violation'] = capital_used - self.constraints.capital_budget
        
        # Check margin constraint
        margin_used = portfolio.margin_used(self.constraints.margin_per_contract)
        if margin_used > self.constraints.margin_cap:
            violations['margin_violation'] = margin_used - self.constraints.margin_cap
        
        # Check CVaR constraint
        cvar = risk_metrics['cvar']
        if cvar < self.constraints.cvar_limit:
            violations['cvar_violation'] = self.constraints.cvar_limit - cvar
        
        constraints_satisfied = all(v == 0.0 for v in violations.values())
        
        return constraints_satisfied, violations
    
    def _create_empty_result(self, status: str) -> OptimizationResult:
        """Create empty result for failed optimization."""
        return OptimizationResult(
            portfolio=Portfolio(positions=[]),
            objective_value=0.0,
            optimization_status=status,
            constraints_satisfied=False,
            portfolio_cvar=0.0,
            portfolio_std=0.0
        )

class GreedyOptimizer:
    """Heuristic greedy optimizer for faster approximate solutions."""
    
    def __init__(self, constraints: OptimizationConstraints):
        self.constraints = constraints
    
    def optimize(
        self,
        contracts: List[OptionContract],
        underlying_volatility: float = 0.20,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        """
        Solve using greedy heuristic approach.
        
        Strategy:
        1. Calculate "efficiency" score for each contract (expected return per unit risk/constraint)
        2. Greedily add contracts with highest efficiency while respecting constraints
        3. Try to balance delta/gamma by adding offsetting positions
        """
        if len(contracts) == 0:
            return self._create_empty_result("No contracts provided")
        
        # Ensure expected returns are calculated
        if any(c.expected_return is None for c in contracts):
            contracts = calculate_expected_returns(contracts)
        
        # Initialize portfolio
        portfolio = Portfolio(positions=[])
        
        # Calculate efficiency scores for each contract
        efficiency_scores = self._calculate_efficiency_scores(contracts)
        
        # Sort contracts by efficiency (best first)
        sorted_indices = np.argsort(efficiency_scores)[::-1]
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            improvement_made = False
            
            # Try adding each contract in order of efficiency
            for idx in sorted_indices:
                contract = contracts[idx]
                
                # Try adding 1 contract (long position)
                if self._try_add_position(portfolio, contract, 1, underlying_volatility):
                    improvement_made = True
                    continue
                
                # Try adding -1 contract (short position) if expected return is positive when short
                if contract.expected_return > 0:  # Overpriced option, good to short
                    if self._try_add_position(portfolio, contract, -1, underlying_volatility):
                        improvement_made = True
                        continue
            
            # If no improvement made, try to optimize existing positions
            if not improvement_made:
                improvement_made = self._optimize_existing_positions(portfolio, contracts, underlying_volatility)
            
            # If still no improvement, break
            if not improvement_made:
                break
        
        # Calculate final risk metrics
        risk_metrics = calculate_portfolio_risk_metrics(
            portfolio, underlying_volatility, self.constraints.cvar_confidence
        )
        
        # Check constraints
        constraints_satisfied, violations = self._check_constraints(portfolio, risk_metrics)
        
        return OptimizationResult(
            portfolio=portfolio,
            objective_value=portfolio.expected_return,
            optimization_status=f"Greedy solution after {iteration} iterations",
            constraints_satisfied=constraints_satisfied,
            portfolio_cvar=risk_metrics['cvar'],
            portfolio_std=risk_metrics['pnl_volatility'],
            **violations
        )
    
    def _calculate_efficiency_scores(self, contracts: List[OptionContract]) -> np.ndarray:
        """Calculate efficiency score for each contract (expected return per unit of constraint usage)."""
        scores = []
        
        for contract in contracts:
            # Expected return per contract
            expected_return = contract.expected_return or 0.0
            
            # "Cost" in terms of constraint usage
            delta_cost = abs(contract.delta)
            gamma_cost = abs(contract.gamma)
            capital_cost = max(0, contract.market_premium * contract.contract_multiplier)  # Only for long positions
            margin_cost = self.constraints.margin_per_contract if expected_return > 0 else 0  # Only for short positions
            
            # Total "cost" (normalized)
            total_cost = (delta_cost / self.constraints.delta_tolerance + 
                         gamma_cost / self.constraints.gamma_tolerance +
                         capital_cost / self.constraints.capital_budget +
                         margin_cost / self.constraints.margin_cap)
            
            # Efficiency = return per unit cost
            if total_cost > 0:
                efficiency = abs(expected_return) / total_cost
            else:
                efficiency = abs(expected_return)
            
            # Prefer contracts with expected return in the direction we want to trade
            if expected_return > 0:  # Good to short (overpriced)
                efficiency *= 1.0
            elif expected_return < 0:  # Good to buy (underpriced)
                efficiency *= 1.0
            else:
                efficiency = 0.0
            
            scores.append(efficiency)
        
        return np.array(scores)
    
    def _try_add_position(
        self, 
        portfolio: Portfolio, 
        contract: OptionContract, 
        quantity: int,
        underlying_volatility: float
    ) -> bool:
        """Try to add a position to the portfolio if constraints allow."""
        # Create temporary portfolio with new position
        temp_portfolio = Portfolio(positions=portfolio.positions.copy())
        temp_portfolio.add_position(contract, quantity)
        
        # Check if constraints are satisfied
        if self._satisfies_constraints(temp_portfolio, underlying_volatility):
            # Add to actual portfolio
            portfolio.add_position(contract, quantity)
            return True
        
        return False
    
    def _satisfies_constraints(self, portfolio: Portfolio, underlying_volatility: float) -> bool:
        """Check if portfolio satisfies all constraints."""
        # Delta constraint
        if abs(portfolio.total_delta) > self.constraints.delta_tolerance:
            return False
        
        # Gamma constraint
        if abs(portfolio.total_gamma) > self.constraints.gamma_tolerance:
            return False
        
        # Capital constraint
        if portfolio.capital_used > self.constraints.capital_budget:
            return False
        
        # Margin constraint
        margin_used = portfolio.margin_used(self.constraints.margin_per_contract)
        if margin_used > self.constraints.margin_cap:
            return False
        
        # CVaR constraint (simplified check)
        cvar_satisfied, _ = check_cvar_constraint(
            portfolio, self.constraints.cvar_limit, underlying_volatility, self.constraints.cvar_confidence
        )
        if not cvar_satisfied:
            return False
        
        return True
    
    def _optimize_existing_positions(
        self, 
        portfolio: Portfolio, 
        contracts: List[OptionContract],
        underlying_volatility: float
    ) -> bool:
        """Try to optimize existing positions by adjusting quantities or adding offsetting positions."""
        improvement_made = False
        
        # Try to add small offsetting positions to improve neutrality
        current_delta = portfolio.total_delta
        current_gamma = portfolio.total_gamma
        
        if abs(current_delta) > 0.1 or abs(current_gamma) > 0.05:
            # Find contracts that could help offset delta/gamma
            for contract in contracts:
                # Calculate how many contracts needed to offset delta
                if abs(contract.delta) > 0.01:
                    offset_quantity = -int(current_delta / contract.delta)
                    if abs(offset_quantity) <= 5:  # Limit adjustment size
                        if self._try_add_position(portfolio, contract, offset_quantity, underlying_volatility):
                            improvement_made = True
                            break
        
        return improvement_made
    
    def _check_constraints(self, portfolio: Portfolio, risk_metrics: dict) -> Tuple[bool, dict]:
        """Check constraints (same logic as MILP optimizer)."""
        optimizer = MILPOptimizer(self.constraints)
        return optimizer._check_constraints(portfolio, risk_metrics)
    
    def _create_empty_result(self, status: str) -> OptimizationResult:
        """Create empty result for failed optimization."""
        return OptimizationResult(
            portfolio=Portfolio(positions=[]),
            objective_value=0.0,
            optimization_status=status,
            constraints_satisfied=False,
            portfolio_cvar=0.0,
            portfolio_std=0.0
        )

def compare_optimizers(
    contracts: List[OptionContract],
    constraints: OptimizationConstraints,
    underlying_volatility: float = 0.20
) -> Dict[str, OptimizationResult]:
    """
    Run both optimizers and compare results.
    
    Returns:
        Dictionary with results from both optimizers
    """
    results = {}
    
    # MILP Optimizer
    milp_optimizer = MILPOptimizer(constraints)
    results['milp'] = milp_optimizer.optimize(contracts, underlying_volatility)
    
    # Greedy Optimizer
    greedy_optimizer = GreedyOptimizer(constraints)
    results['greedy'] = greedy_optimizer.optimize(contracts, underlying_volatility)
    
    return results 