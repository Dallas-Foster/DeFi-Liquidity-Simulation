# rl_utils.py

import numpy as np

class SimpleQPolicy:
    """
    A minimal Q-learning approach for the RLLiquidityProvider:
    - State: (discretized price difference, step mod 10, etc.)
    - Actions: [0: do nothing, 1: add liquidity, 2: remove liquidity]
    """

    def __init__(self,
                 num_price_buckets=5,
                 num_time_buckets=10,
                 alpha=0.1,      # learning rate
                 gamma=0.95,     # discount factor
                 epsilon=0.1,    # exploration rate
                 random_seed=None):
        """
        If 'random_seed' is None, we do not seed the global RNG.
        If 'random_seed' is an integer, we call np.random.seed(...) so that
        this policy's behavior is reproducible.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.num_price_buckets = num_price_buckets
        self.num_time_buckets = num_time_buckets
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table shape = (price_bucket, time_bucket, action)
        self.q_table = np.zeros((self.num_price_buckets, self.num_time_buckets, 3))

        self.last_state = None
        self.last_action = None

    def _discretize_state(self, env_state):
        # Discretize the difference between ref_price and amm_price
        price_diff = env_state["reference_price"] - env_state["amm_price"]
        # For demonstration, bucket from -X to +X in 0.02 increments
        bucket_size = 0.02
        bucket = int((price_diff + self.num_price_buckets * bucket_size) // bucket_size)
        bucket = max(0, min(self.num_price_buckets - 1, bucket))

        time_bucket = env_state["step"] % self.num_time_buckets
        return (bucket, time_bucket)

    def select_action(self, env_state):
        # Convert env_state to discrete
        s = self._discretize_state(env_state)

        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            a = np.random.randint(0, 3)
        else:
            a = np.argmax(self.q_table[s[0], s[1], :])

        # Store for learning
        self.last_state = s
        self.last_action = a

        if a == 0:
            # do nothing
            return {
                "type": "none",
                "amount_tokenA": 0.0,
                "amount_tokenB": 0.0
            }
        elif a == 1:
            # add liquidity
            return {
                "type": "add_liquidity",
                "amount_tokenA": 10.0,
                "amount_tokenB": 10.0
            }
        else:
            # remove some fraction of liquidity
            return {
                "type": "remove_liquidity",
                "amount_tokenA": 0.3,  # fraction
                "amount_tokenB": 0.3
            }

    def update_q(self, reward, new_env_state, done=False):
        """
        Called after each step to update Q-table via Q-learning.
        If 'done', treat next state's value as 0 (no future rewards).
        """
        if self.last_state is None:
            return

        s = self.last_state
        a = self.last_action

        if done:
            target = reward
        else:
            s_next = self._discretize_state(new_env_state)
            target = reward + self.gamma * np.max(self.q_table[s_next[0], s_next[1], :])

        self.q_table[s[0], s[1], a] += self.alpha * (target - self.q_table[s[0], s[1], a])

        # Clear the stored state/action
        self.last_state = None
        self.last_action = None
