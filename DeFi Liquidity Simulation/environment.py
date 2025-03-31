"""
environment.py

Orchestrates the simulation: it holds the AMM, the agents, and the external
reference price data. At each step, it updates the environment based on agent actions.
"""

import numpy as np

class MarketEnvironment:
    def __init__(self, amm, agents, reference_prices):
        """
        amm (BaseAMM): An AMM instance.
        agents (list): A list of agent instances.
        reference_prices (np.array): The external "true" or reference market price.
        """
        self.amm = amm
        self.agents = agents
        self.reference_prices = reference_prices
        self.num_steps = len(reference_prices)
        self.current_step = 0
        self.rewards = {}
        for agent in agents:
            self.rewards[agent.name] = 0.0

        self.total_volume_tokenA = 0.0
        self.total_volume_tokenB = 0.0

    def reset(self):
        self.current_step = 0
        for agent in self.agents:
            self.rewards[agent.name] = 0.0
        self.total_volume_tokenA = 0.0
        self.total_volume_tokenB = 0.0

    def step(self):
        if self.current_step >= self.num_steps:
            return False

        current_price = self.reference_prices[self.current_step]
        amm_price = self.amm.get_price()

        env_state = {
            "amm_price": amm_price,
            "reference_price": current_price,
            "step": self.current_step
        }

        # Collect agent actions
        actions = {}
        for agent in self.agents:
            actions[agent.name] = agent.act(env_state)

        # Execute actions
        for agent in self.agents:
            action = actions[agent.name]

            if action["type"] == "trade":
                direction = action.get("direction", "buy")
                amount_in = action.get("amount", 0.0)
                if amount_in > 0:
                    tokens_received = self.amm.swap(amount_in, direction)
                    if direction == "buy":
                        # cost in "B" is amount_in
                        cost_ref = amount_in
                        value_ref = tokens_received * current_price
                        self.rewards[agent.name] += (value_ref - cost_ref)
                        self.total_volume_tokenB += amount_in
                    else:
                        # direction == "sell"
                        cost_ref = amount_in * current_price
                        value_ref = tokens_received
                        self.rewards[agent.name] += (value_ref - cost_ref)
                        self.total_volume_tokenA += amount_in

            elif action["type"] == "add_liquidity":
                amountA = action["amount_tokenA"]
                amountB = action["amount_tokenB"]
                if (amountA > 0 and amountB > 0):
                    self.amm.add_liquidity(amountA, amountB)

            elif action["type"] == "remove_liquidity":
                fractionA = action["amount_tokenA"]
                fractionB = action["amount_tokenB"]
                fraction = min(fractionA, fractionB)  # or average them
                result = self.amm.remove_liquidity(fraction)
                if result is not None:
                    outA, outB = result
                    # approximate reward in reference terms
                    self.rewards[agent.name] += outA * current_price + outB

        self.current_step += 1
        return True

    def run_simulation(self):
        self.reset()
        while self.step():
            pass

    def get_rewards(self):
        return self.rewards

    def get_current_step(self):
        return self.current_step
