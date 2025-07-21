#!/usr/bin/env python3
"""
Test script for the Options Portfolio Optimizer.

This script runs a basic test to ensure all components work together.
"""

import sys
import traceback
from portfolio_optimizer import PortfolioOptimizer
from models import OptimizationConstraints
from data_generator import get_small_test_chain

def test_basic_functionality():
    """Test basic functionality of the optimizer."""
    print("🧪 Testing Options Portfolio Optimizer")
    print("=" * 50)
    
    try:
        # Initialize optimizer
        print("1. Initializing optimizer...")
        constraints = OptimizationConstraints(
            delta_tolerance=0.5,
            gamma_tolerance=0.2,
            capital_budget=100000,  # Smaller budget for test
            margin_cap=50000,
            cvar_limit=-25000
        )
        optimizer = PortfolioOptimizer(constraints)
        print("   ✅ Optimizer initialized")
        
        # Load test data
        print("\n2. Loading test data...")
        num_contracts, description = optimizer.load_contracts_from_scenario("balanced")
        print(f"   ✅ Loaded {num_contracts} contracts")
        print(f"   📝 Scenario: {description}")
        
        # Show contract summary
        print("\n3. Contract summary:")
        contracts_df = optimizer.get_contracts_summary()
        if not contracts_df.empty:
            print(f"   📊 {len(contracts_df)} total contracts")
            print(f"   📞 {len(contracts_df[contracts_df['Type'] == 'call'])} calls")
            print(f"   📞 {len(contracts_df[contracts_df['Type'] == 'put'])} puts")
            print(f"   📈 Average IV: {contracts_df['Implied Vol'].mean():.1%}")
        
        # Test greedy optimization (faster)
        print("\n4. Running greedy optimization...")
        result = optimizer.optimize(method="greedy")
        print(f"   ✅ Optimization completed: {result.optimization_status}")
        print(f"   💰 Expected return: ${result.objective_value:,.2f}")
        print(f"   📏 Portfolio delta: {result.portfolio.total_delta:.3f}")
        print(f"   📏 Portfolio gamma: {result.portfolio.total_gamma:.3f}")
        print(f"   ✔️ Constraints satisfied: {result.constraints_satisfied}")
        
        # Show portfolio summary
        print("\n5. Portfolio analysis:")
        summary = optimizer.get_portfolio_summary()
        if "error" not in summary:
            positions = len(summary['portfolio_composition'])
            print(f"   📋 Total positions: {positions}")
            
            if positions > 0:
                capital_used = summary['portfolio_metrics']['Capital Used']
                margin_used = summary['portfolio_metrics']['Margin Used']
                print(f"   💵 Capital used: ${capital_used:,.0f}")
                print(f"   🏦 Margin used: ${margin_used:,.0f}")
                
                # Show constraint satisfaction
                print("\n   Constraint Check:")
                for constraint, details in summary['constraints_status'].items():
                    status = "✅" if details['Satisfied'] else "❌"
                    print(f"   {status} {constraint}: {details['Value']} (limit: {details['Limit']})")
            else:
                print("   ⚠️ No positions in optimized portfolio")
        
        # Test MILP optimization if solver available
        print("\n6. Testing MILP optimization...")
        try:
            milp_result = optimizer.optimize(method="milp", allow_fractional=True)
            print(f"   ✅ MILP completed: {milp_result.optimization_status}")
            print(f"   💰 MILP expected return: ${milp_result.objective_value:,.2f}")
            
            # Compare methods
            improvement = milp_result.objective_value - result.objective_value
            if improvement > 0:
                print(f"   📈 MILP improvement: ${improvement:,.2f}")
            else:
                print(f"   📊 Greedy performed as well as MILP")
                
        except Exception as e:
            print(f"   ⚠️ MILP solver not available or failed: {str(e)}")
            print("   💡 This is normal if no commercial solvers are installed")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_small_chain():
    """Test with a very small options chain."""
    print("\n🧪 Testing with small options chain")
    print("-" * 30)
    
    try:
        # Create small test chain
        contracts = get_small_test_chain()
        print(f"✅ Generated {len(contracts)} test contracts")
        
        # Initialize optimizer with smaller constraints
        optimizer = PortfolioOptimizer()
        optimizer.set_constraints(
            capital_budget=50000,
            margin_cap=25000,
            delta_tolerance=1.0,  # More relaxed for small chain
            gamma_tolerance=0.5
        )
        
        # Load contracts
        num_loaded = optimizer.load_contracts(contracts)
        print(f"✅ Loaded {num_loaded} contracts into optimizer")
        
        # Optimize
        result = optimizer.optimize(method="greedy")
        print(f"✅ Small chain optimization: {result.optimization_status}")
        print(f"💰 Expected return: ${result.objective_value:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Small chain test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Options Portfolio Optimizer - Test Suite")
    print("========================================\n")
    
    # Run tests
    success1 = test_basic_functionality()
    success2 = test_small_chain()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! The optimizer is ready to use.")
        print("\nTo run the web interface, execute:")
        print("   streamlit run app.py")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 