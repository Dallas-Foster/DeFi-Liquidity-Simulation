import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from amms import ConstantProductAMM
from agents import ArbitrageBot, RandomTrader, BasicLiquidityProvider, RLLiquidityProvider
from environment import MarketEnvironment
from data_utils import generate_synthetic_price_series
from rl_utils import SimpleQPolicy
from visualizations import plot_price_series, plot_rewards, plot_amm_vs_ref


def train_rl_agent(num_epochs=5, steps_per_epoch=300):
    """
    Trains the RL agent by running multiple short simulations.
    No seeds = fully random every run (assuming fresh interpreter).
    """
    policy = SimpleQPolicy()

    for _ in range(num_epochs):
        # Generate random reference prices (no seed argument).
        reference_prices = generate_synthetic_price_series(
            num_steps=steps_per_epoch,
            start_price=1.0,
            drift=0.0,
            volatility=0.02
        )

        amm = ConstantProductAMM(fee_rate=0.003)
        rl_agent = RLLiquidityProvider("RL_LP", policy=policy)
        random_trader = RandomTrader("RandomTrader")
        env = MarketEnvironment(amm, [rl_agent, random_trader], reference_prices)

        env.reset()
        while env.step():
            reward = env.get_rewards()["RL_LP"]
            done = (env.get_current_step() >= steps_per_epoch)
            new_state = {
                "amm_price": amm.get_price(),
                "reference_price": reference_prices[min(env.get_current_step(), steps_per_epoch-1)],
                "step": env.get_current_step()
            }
            policy.update_q(reward, new_state, done=done)
    return policy


def run_final_simulation(policy, steps=300):
    """
    Runs a final simulation with multiple agents, again with random reference prices.
    """
    reference_prices = generate_synthetic_price_series(
        num_steps=steps,
        start_price=1.2,
        drift=0.0,
        volatility=0.02
    )

    amm = ConstantProductAMM(fee_rate=0.003)
    rl_agent = RLLiquidityProvider("RL_LP", policy=policy)
    arb_bot = ArbitrageBot("ArbBot", threshold=0.002, max_trade_size=10.0)
    random_trader = RandomTrader("RandomTrader", max_trade_size=5.0)
    basic_lp = BasicLiquidityProvider("BasicLP", initial_liquidity=500.0)

    agents = [rl_agent, arb_bot, random_trader, basic_lp]
    env = MarketEnvironment(amm, agents, reference_prices)
    env.run_simulation()

    rewards = env.get_rewards()
    return reference_prices, env, rewards


def main():
    print("DeFi Liquidity Simulator - FULLY RANDOM each run!")
    # This is just a random check to confirm that each run prints a new random number:
    print("Random check:", np.random.rand())

    # Train RL agent
    policy = train_rl_agent(num_epochs=5, steps_per_epoch=300)

    # Run final simulation
    ref_prices, env, final_rewards = run_final_simulation(policy, steps=300)
    print("Final Rewards:", final_rewards)

    # Visualize
    plot_price_series(ref_prices)

    # Dummy AMM price trace for illustration
    amm_prices = []
    dummy_amm = ConstantProductAMM()
    amm_prices.append(dummy_amm.get_price())
    for p in ref_prices:
        dummy_amm.add_liquidity(1.0, p)
        amm_prices.append(dummy_amm.get_price())

    plot_amm_vs_ref(amm_prices, ref_prices)

    agent_names = list(final_rewards.keys())
    plot_rewards(agent_names, final_rewards)


if __name__ == "__main__":
    main()
