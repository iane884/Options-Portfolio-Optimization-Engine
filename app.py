import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

from portfolio_optimizer import PortfolioOptimizer
from models import OptimizationConstraints
from data_generator import get_small_test_chain

def main():
    st.set_page_config(
        page_title="Options Portfolio Optimizer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Options Portfolio Optimizer")
    st.markdown("**Maximize expected return while maintaining delta/gamma neutrality and risk constraints**")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PortfolioOptimizer()
    
    optimizer = st.session_state.optimizer
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Predefined Scenario", "Small Test Chain", "Upload CSV (Future)"]
    )
    
    if data_source == "Predefined Scenario":
        scenario = st.sidebar.selectbox(
            "Select scenario:",
            ["balanced", "high_vol", "low_vol", "skewed"]
        )
        
        if st.sidebar.button("Load Scenario Data"):
            try:
                num_contracts, description = optimizer.load_contracts_from_scenario(scenario)
                st.session_state.data_loaded = True
                st.sidebar.success(f"Loaded {num_contracts} contracts")
                st.sidebar.info(description)
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
    
    elif data_source == "Small Test Chain":
        if st.sidebar.button("Load Test Data"):
            try:
                contracts = get_small_test_chain()
                num_contracts = optimizer.load_contracts(contracts)
                st.session_state.data_loaded = True
                st.sidebar.success(f"Loaded {num_contracts} test contracts")
            except Exception as e:
                st.sidebar.error(f"Error loading test data: {str(e)}")
    
    # Optimization constraints
    st.sidebar.subheader("Constraints")
    
    delta_tolerance = st.sidebar.number_input(
        "Delta Tolerance (±)", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.5, 
        step=0.1,
        help="Maximum absolute portfolio delta"
    )
    
    gamma_tolerance = st.sidebar.number_input(
        "Gamma Tolerance (±)", 
        min_value=0.05, 
        max_value=1.0, 
        value=0.2, 
        step=0.05,
        help="Maximum absolute portfolio gamma"
    )
    
    capital_budget = st.sidebar.number_input(
        "Capital Budget ($)", 
        min_value=10000, 
        max_value=10000000, 
        value=500000, 
        step=10000,
        help="Maximum capital to deploy"
    )
    
    margin_cap = st.sidebar.number_input(
        "Margin Cap ($)", 
        min_value=10000, 
        max_value=5000000, 
        value=100000, 
        step=10000,
        help="Maximum margin usage"
    )
    
    cvar_limit = st.sidebar.number_input(
        "CVaR Limit ($)", 
        min_value=-500000, 
        max_value=-1000, 
        value=-75000, 
        step=1000,
        help="Maximum acceptable CVaR (negative value)"
    )
    
    underlying_volatility = st.sidebar.slider(
        "Underlying Volatility", 
        min_value=0.05, 
        max_value=0.50, 
        value=0.20, 
        step=0.01,
        help="Annualized volatility of underlying asset"
    )
    
    # Update constraints
    optimizer.set_constraints(
        delta_tolerance=delta_tolerance,
        gamma_tolerance=gamma_tolerance,
        capital_budget=capital_budget,
        margin_cap=margin_cap,
        cvar_limit=cvar_limit
    )
    optimizer.set_underlying_volatility(underlying_volatility)
    
    # Optimization method
    st.sidebar.subheader("Optimization")
    
    method = st.sidebar.selectbox(
        "Optimization Method",
        ["milp", "greedy", "both"],
        help="MILP: Exact solution, Greedy: Fast heuristic, Both: Compare methods"
    )
    
    allow_fractional = st.sidebar.checkbox(
        "Allow Fractional Contracts",
        value=False,
        help="Allow fractional contract quantities (MILP only)"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Contract Data")
        
        # Show loaded contracts
        if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
            contracts_df = optimizer.get_contracts_summary()
            if not contracts_df.empty:
                st.dataframe(contracts_df, use_container_width=True, height=300)
                
                # Show summary statistics
                st.subheader("Data Summary")
                col1a, col1b, col1c, col1d = st.columns(4)
                
                with col1a:
                    st.metric("Total Contracts", len(contracts_df))
                with col1b:
                    calls = len(contracts_df[contracts_df['Type'] == 'call'])
                    st.metric("Calls", calls)
                with col1c:
                    puts = len(contracts_df[contracts_df['Type'] == 'put'])
                    st.metric("Puts", puts)
                with col1d:
                    avg_iv = contracts_df['Implied Vol'].mean()
                    st.metric("Avg Implied Vol", f"{avg_iv:.1%}")
        else:
            st.info("Please load contract data from the sidebar to begin optimization.")
    
    with col2:
        st.header("Quick Actions")
        
        if st.button("🚀 Run Optimization", type="primary", disabled=not hasattr(st.session_state, 'data_loaded')):
            if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
                with st.spinner("Running optimization..."):
                    try:
                        result = optimizer.optimize(method=method, allow_fractional=allow_fractional)
                        st.session_state.optimization_result = result
                        st.success("Optimization completed!")
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
            else:
                st.error("Please load data first")
        
        if st.button("📊 Show Results", disabled=not hasattr(st.session_state, 'optimization_result')):
            st.session_state.show_results = True
        
        if st.button("💾 Export Portfolio", disabled=not hasattr(st.session_state, 'optimization_result')):
            try:
                filename = optimizer.export_portfolio_to_csv()
                st.success(f"Portfolio exported to {filename}")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # Results section
    if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
        if hasattr(st.session_state, 'optimization_result'):
            st.header("Optimization Results")
            
            result = st.session_state.optimization_result
            summary = optimizer.get_portfolio_summary(result)
            
            if "error" in summary:
                st.error(summary["error"])
            else:
                # High-level metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Expected Return", 
                        f"${summary['portfolio_metrics']['Expected Return']:,.0f}"
                    )
                with col2:
                    st.metric(
                        "Portfolio Delta", 
                        f"{summary['portfolio_metrics']['Portfolio Delta']:.3f}"
                    )
                with col3:
                    st.metric(
                        "Portfolio Gamma", 
                        f"{summary['portfolio_metrics']['Portfolio Gamma']:.3f}"
                    )
                with col4:
                    constraints_satisfied = "✅" if summary['constraints_satisfied'] else "❌"
                    st.metric("Constraints", constraints_satisfied)
                
                # Detailed results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Positions", "Risk Metrics", "Constraints Check", "Scenario Analysis"])
                
                with tab1:
                    st.subheader("Portfolio Positions")
                    if summary['portfolio_composition']:
                        positions_df = pd.DataFrame(summary['portfolio_composition'])
                        st.dataframe(positions_df, use_container_width=True)
                        
                        # Position summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            long_positions = len([p for p in summary['portfolio_composition'] if p['Quantity'] > 0])
                            st.metric("Long Positions", long_positions)
                        with col2:
                            short_positions = len([p for p in summary['portfolio_composition'] if p['Quantity'] < 0])
                            st.metric("Short Positions", short_positions)
                        with col3:
                            net_premium = summary['portfolio_metrics']['Net Premium']
                            st.metric("Net Premium", f"${net_premium:,.0f}")
                    else:
                        st.info("No positions in optimized portfolio")
                
                with tab2:
                    st.subheader("Risk Metrics")
                    
                    risk_cols = st.columns(2)
                    with risk_cols[0]:
                        st.metric("CVaR (95%)", f"${summary['risk_metrics']['cvar']:,.0f}")
                        st.metric("VaR (95%)", f"${summary['risk_metrics']['var']:,.0f}")
                        st.metric("P&L Volatility", f"${summary['risk_metrics']['pnl_volatility']:,.0f}")
                    
                    with risk_cols[1]:
                        st.metric("Sharpe Ratio", f"{summary['risk_metrics']['sharpe_ratio']:.3f}")
                        st.metric("Portfolio Vega", f"{summary['portfolio_metrics']['Portfolio Vega']:,.0f}")
                        st.metric("Time Horizon", f"{summary['risk_metrics']['time_horizon_days']} day(s)")
                
                with tab3:
                    st.subheader("Constraint Satisfaction")
                    
                    constraints_data = []
                    for constraint, details in summary['constraints_status'].items():
                        status = "✅ Satisfied" if details['Satisfied'] else "❌ Violated"
                        constraints_data.append({
                            'Constraint': constraint,
                            'Value': details['Value'],
                            'Limit': details['Limit'],
                            'Status': status
                        })
                    
                    constraints_df = pd.DataFrame(constraints_data)
                    st.dataframe(constraints_df, use_container_width=True)
                
                with tab4:
                    st.subheader("Scenario Analysis")
                    
                    if st.button("Run Scenario Analysis"):
                        with st.spinner("Analyzing scenarios..."):
                            scenario_analysis = optimizer.analyze_scenario_sensitivity()
                            
                            if "error" not in scenario_analysis:
                                scenarios_df = pd.DataFrame(scenario_analysis['scenarios'])
                                
                                # Create a pivot table for better visualization
                                pivot_df = scenarios_df.pivot_table(
                                    values='Estimated P&L',
                                    index='Price Change %',
                                    columns='Vol Change %',
                                    aggfunc='first'
                                )
                                
                                st.subheader("P&L Heatmap")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto')
                                
                                # Set labels
                                ax.set_xticks(range(len(pivot_df.columns)))
                                ax.set_xticklabels([f"{x:.0f}%" for x in pivot_df.columns])
                                ax.set_yticks(range(len(pivot_df.index)))
                                ax.set_yticklabels([f"{y:.0f}%" for y in pivot_df.index])
                                
                                ax.set_xlabel('Volatility Change')
                                ax.set_ylabel('Price Change')
                                ax.set_title('Portfolio P&L Under Different Scenarios')
                                
                                # Add colorbar
                                plt.colorbar(im, ax=ax, label='P&L ($)')
                                
                                st.pyplot(fig)
                                
                                # Show detailed table
                                st.subheader("Detailed Scenarios")
                                st.dataframe(scenarios_df, use_container_width=True)
                            else:
                                st.error(scenario_analysis["error"])

if __name__ == "__main__":
    main() 