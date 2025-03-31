# DeFi-Liquidity-Simulation
Python-based simulation environment that models a decentralized finance (DeFi) ecosystem using an Automated Market Maker (AMM) similar to Uniswap v2. Supports a variety of agent types, including a reinforcement learning (RL)-based liquidity provider, and simulates market interactions over synthetic price data.

# Project Overview
The simulator consists of the following core components:

Automated Market Maker (AMM): Implements a constant product market-making strategy (X * Y = K).

Agents: Simulated participants including:

**RandomTrader**: Trades randomly to simulate noise.

**ArbitrageBot**: Exploits price discrepancies between the AMM and external market.

**BasicLiquidityProvider**: Adds/removes liquidity passively based on market conditions.

**RLLiquidityProvider**: Uses a simple Q-learning policy to optimize liquidity provisioning.

**RL Agent**: A basic tabular Q-learning agent trained to learn when to provide or remove liquidity for maximizing profit.

Market Environment: Coordinates agent actions, AMM state updates, and reward calculations over a simulated time series of prices.

Synthetic Price Generator: Generates random price movements using geometric Brownian motion.

Visualizations: Matplotlib-based plots to show reference prices, AMM pricing trends, and final agent rewards.
