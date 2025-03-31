import matplotlib.pyplot as plt

def plot_price_series(reference_prices):
    plt.figure()
    plt.title("Reference Price Series")
    plt.xlabel("Timestep")
    plt.ylabel("Price")
    plt.plot(reference_prices)
    plt.show()

def plot_rewards(agent_names, rewards):
    """
    agent_names (list of str)
    rewards (dict): final reward for each agent's name
    """
    plt.figure()
    plt.title("Final Agent Rewards")
    plt.xlabel("Agent")
    plt.ylabel("Reward")
    # For a bar chart, we need x positions
    x = range(len(agent_names))
    y = [rewards[name] for name in agent_names]
    plt.bar(x, y)
    plt.xticks(x, agent_names, rotation=45)
    plt.tight_layout()
    plt.show()

def plot_amm_vs_ref(amm_prices, ref_prices):
    plt.figure()
    plt.title("AMM Price vs Reference Price")
    plt.xlabel("Timestep")
    plt.ylabel("Price")
    plt.plot(ref_prices, label="Reference Price")
    plt.plot(amm_prices, label="AMM Price")
    plt.legend()
    plt.show()
