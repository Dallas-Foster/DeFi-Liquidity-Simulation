"""
agents.py

Defines the different agent classes (ArbitrageBot, RandomTrader,
BasicLiquidityProvider, RLLiquidityProvider) that will act in the simulation.
"""

import numpy as np

class BaseAgent:
    """
    Abstract base class for agents. Agents must implement the `act` method,
    which returns a dictionary describing the action.
    """
    def __init__(self, name):
        self.name = name

    def act(self, env_state):
        raise NotImplementedError


class ArbitrageBot(BaseAgent):
    """
    An agent that checks if there's a significant price discrepancy between
    the AMM price and the reference market price. If profitable, it trades
    to exploit the discrepancy.
    """
    def __init__(self, name, threshold=0.001, max_trade_size=10.0):
        super().__init__(name)
        self.threshold = threshold  # E.g., 0.1% threshold
        self.max_trade_size = max_trade_size

    def act(self, env_state):
        amm_price = env_state["amm_price"]
        ref_price = env_state["reference_price"]

        # If amm_price is near zero, skip to avoid divide-by-zero in price_diff / amm_price
        if amm_price < 1e-12:
            return {"type": "none", "amount": 0.0}

        price_diff = ref_price - amm_price
        relative_diff = price_diff / amm_price

        action = {
            "type": "none",
            "amount": 0.0
        }

        # If reference price > AMM price by threshold => buy from AMM cheaply
        if relative_diff > self.threshold:
            action["type"] = "trade"
            action["amount"] = min(self.max_trade_size, abs(price_diff) * 10)
            action["direction"] = "buy"  # Buy Token A, spend Token B
        # If AMM price > reference price by threshold => sell to AMM
        elif -relative_diff > self.threshold:
            action["type"] = "trade"
            action["amount"] = min(self.max_trade_size, abs(price_diff) * 10)
            action["direction"] = "sell"  # Sell Token A, receive Token B

        return action


class RandomTrader(BaseAgent):
    """
    Executes random trades with random directions (buy/sell).
    Useful for adding background noise into the simulation.
    """
    def __init__(self, name, max_trade_size=5.0, trade_prob=0.5):
        super().__init__(name)
        self.max_trade_size = max_trade_size
        self.trade_prob = trade_prob

    def act(self, env_state):
        action = {
            "type": "none",
            "amount": 0.0
        }
        # Decide randomly whether to trade
        if np.random.rand() < self.trade_prob:
            action["type"] = "trade"
            action["amount"] = np.random.rand() * self.max_trade_size
            action["direction"] = np.random.choice(["buy", "sell"])
        return action


class BasicLiquidityProvider(BaseAgent):
    """
    A basic liquidity provider that provides a fixed amount of liquidity
    at the start and does not rebalance unless an extreme condition hits.
    """
    def __init__(self, name, initial_liquidity=1000.0, remove_threshold=0.2):
        super().__init__(name)
        self.initial_liquidity = initial_liquidity
        self.remove_threshold = remove_threshold
        self.has_provided = False

    def act(self, env_state):
        action = {
            "type": "none",
            "amount_tokenA": 0.0,
            "amount_tokenB": 0.0
        }

        if not self.has_provided:
            # Provide liquidity once at the start
            action["type"] = "add_liquidity"
            ref_price = env_state["reference_price"]
            amount_B = self.initial_liquidity * ref_price
            action["amount_tokenA"] = self.initial_liquidity
            action["amount_tokenB"] = amount_B
            self.has_provided = True
        else:
            # Possibly remove liquidity if price has changed more than threshold
            amm_price = env_state["amm_price"]
            ref_price = env_state["reference_price"]
            if ref_price > 1e-12:  # avoid zero-division
                price_diff = abs(amm_price - ref_price) / ref_price
                if price_diff > self.remove_threshold:
                    action["type"] = "remove_liquidity"
                    # remove half the liquidity
                    action["amount_tokenA"] = 0.5  # fraction
                    action["amount_tokenB"] = 0.5
        return action


class RLLiquidityProvider(BaseAgent):
    """
    A liquidity provider that uses an external RL policy (Q-table or network)
    to decide how much liquidity to add/remove each step.
    """
    def __init__(self, name, policy):
        super().__init__(name)
        self.policy = policy

    def act(self, env_state):
        # Query policy for an action
        action = self.policy.select_action(env_state)
        return action
