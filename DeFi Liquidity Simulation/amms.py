"""
amms.py

Implements a simple Constant Product AMM (like Uniswap V2) and a base class
for potential extensions (e.g., stable-swap, hybrid AMM).
"""

class BaseAMM:
    """
    Base class for an AMM.
    """
    def __init__(self, fee_rate=0.003):
        self.fee_rate = fee_rate

    def add_liquidity(self, amount_tokenA, amount_tokenB):
        raise NotImplementedError

    def remove_liquidity(self, liquidity_shares):
        raise NotImplementedError

    def swap(self, amount_in, direction):
        raise NotImplementedError

    def get_price(self):
        raise NotImplementedError

    def total_liquidity_shares(self):
        raise NotImplementedError


class ConstantProductAMM(BaseAMM):
    """
    Uniswap v2–style Constant Product Automated Market Maker.
    X * Y = K, where X = reserve of Token A, Y = reserve of Token B.
    """
    def __init__(self, fee_rate=0.003):
        super().__init__(fee_rate=fee_rate)
        self.reserveA = 0.0
        self.reserveB = 0.0
        self.K = 0.0
        self.liquidity_shares = 0.0

    def add_liquidity(self, amount_tokenA, amount_tokenB):
        """
        Provide liquidity in proportion to the existing ratio of reserves.
        If the pool is empty or effectively empty, treat it as new.
        """
        # If either reserve is near zero, treat as new pool
        if self.reserveA < 1e-12 or self.reserveB < 1e-12:
            self.reserveA = amount_tokenA
            self.reserveB = amount_tokenB
            self.K = self.reserveA * self.reserveB
            # For simplicity: mint shares = sum of tokens
            self.liquidity_shares += (amount_tokenA + amount_tokenB)
            return True

        # Otherwise, deposit in the ratio of existing reserves
        ratioA = amount_tokenA / self.reserveA if self.reserveA != 0 else float('inf')
        ratioB = amount_tokenB / self.reserveB if self.reserveB != 0 else float('inf')

        # For realism, you'd revert or adjust if ratioA != ratioB, but let's proceed:
        self.reserveA += amount_tokenA
        self.reserveB += amount_tokenB
        self.K = self.reserveA * self.reserveB

        # Avoid dividing by zero if self.reserveA == amount_tokenA
        denominator = (self.reserveA - amount_tokenA)
        if abs(denominator) < 1e-12:
            # If the user is depositing all or if pool was tiny, fallback
            # Just mint a proportion of the existing shares
            shares_minted = amount_tokenA + amount_tokenB
        else:
            shares_minted = (self.liquidity_shares * amount_tokenA) / denominator

        self.liquidity_shares += shares_minted
        return True

    def remove_liquidity(self, fraction):
        """
        Remove fraction of total liquidity shares.
        Prevent removing 100% to avoid emptying the pool entirely.
        """
        # If fraction is too close to 1, limit it
        fraction = min(fraction, 0.999)  # remove up to 99.9%
        if fraction < 0:
            return None

        amountA_out = self.reserveA * fraction
        amountB_out = self.reserveB * fraction

        self.reserveA -= amountA_out
        self.reserveB -= amountB_out

        shares_to_burn = self.liquidity_shares * fraction
        self.liquidity_shares -= shares_to_burn

        # Update K
        self.K = self.reserveA * self.reserveB
        return amountA_out, amountB_out

    def swap(self, amount_in, direction):
        if amount_in <= 0:
            return 0.0

        # If either reserve is near zero, swapping is effectively impossible
        if self.reserveA < 1e-12 or self.reserveB < 1e-12:
            return 0.0

        if direction == "buy":
            # user inputs B, gets A
            amount_in_after_fee = amount_in * (1 - self.fee_rate)
            new_reserveB = self.reserveB + amount_in_after_fee
            # x * y = k => x = K / y
            new_reserveA = self.K / new_reserveB if new_reserveB != 0 else 0
            amount_out = self.reserveA - new_reserveA
            if amount_out < 0:
                amount_out = 0
            self.reserveB = new_reserveB
            self.reserveA = new_reserveA
        else:
            # direction == "sell"
            amount_in_after_fee = amount_in * (1 - self.fee_rate)
            new_reserveA = self.reserveA + amount_in_after_fee
            new_reserveB = self.K / new_reserveA if new_reserveA != 0 else 0
            amount_out = self.reserveB - new_reserveB
            if amount_out < 0:
                amount_out = 0
            self.reserveA = new_reserveA
            self.reserveB = new_reserveB

        self.K = self.reserveA * self.reserveB
        return amount_out

    def get_price(self):
        if self.reserveA < 1e-12:
            return 0.0
        return self.reserveB / self.reserveA

    def total_liquidity_shares(self):
        return self.liquidity_shares
