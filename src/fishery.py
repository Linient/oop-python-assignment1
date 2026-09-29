"""
fishery.py - Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.

Main pieces:
    fibonacci()                 - the old, flawed "stock" baseline
    FishStock                   - logistic growth with harvesting
    PriceModel                  - weekly fish price as a bounded random walk
    RiskAssessor                - turns revenue into a risk class + Value-at-Risk
    weekly_revenue()            - harvest x price
    monte_carlo_annual_revenue()- 1,000 possible years of revenue
"""

# statistics (built in): mean, median, variance, stdev.
import statistics

# numpy: arrays and random numbers.
import numpy as np

KG_PER_TONNE = 1000   # used to turn tonnes of fish into kg


def fibonacci(count: int) -> list[int]:
    """The first `count` Fibonacci numbers: 1, 1, 2, 3, 5, 8, ...

    Each number is the sum of the two before it. Used only as the baseline
    the assignment asks us to criticise.
    """
    if count < 0:
        raise ValueError("count cannot be negative")
    numbers = []
    a, b = 1, 1
    for _ in range(count):
        numbers.append(a)
        a, b = b, a + b      # move one step along the sequence
    return numbers


class FishStock:
    """Fish population (in tonnes) following logistic growth with harvesting.

    Each week:
        N(t+1) = N(t) + r*N(t)*(1 - N(t)/K) - h*N(t)
                        \\_____ growth ____/    \\harvest/
    Attributes:
        growth_rate (r)  - how fast fish reproduce when there is plenty of room
        capacity (K)     - the most fish the lake can support (tonnes)
        initial_stock    - N(0), stock at the start (tonnes)
        harvest_rate (h) - share of the stock caught each week (0.1 = 10%)
    """

    def __init__(self, growth_rate: float = 0.4, capacity: float = 10_000,
                 initial_stock: float = 4_000, harvest_rate: float = 0.1) -> None:
        # (10_000 is just 10000 - Python lets you use _ to make big numbers readable)
        if capacity <= 0:
            raise ValueError("carrying capacity must be positive")
        if initial_stock < 0:
            raise ValueError("stock cannot be negative")
        if not 0 <= harvest_rate <= 1:
            raise ValueError("harvest rate must be between 0 and 1")
        self.growth_rate = growth_rate
        self.capacity = capacity
        self.initial_stock = initial_stock
        self.harvest_rate = harvest_rate

    def __repr__(self) -> str:
        return (f"FishStock(r={self.growth_rate}, K={self.capacity}, "
                f"N0={self.initial_stock}, h={self.harvest_rate})")

    def simulate(self, weeks: int = 52, closed_weeks_per_year: int = 0) -> tuple:
        """Run the model forward week by week.

        closed_weeks_per_year: for the extension - the first N weeks of every
        52-week year have NO fishing (a closed season).

        Returns (stock, harvest):
            stock   - array of length weeks + 1 (includes week 0), tonnes
            harvest - array of length weeks, tonnes caught each week
        """
        stock = np.zeros(weeks + 1)
        harvest = np.zeros(weeks)
        stock[0] = self.initial_stock
        for week in range(weeks):
            n = stock[week]                                   # stock this week
            growth = self.growth_rate * n * (1 - n / self.capacity)
            fishing_is_closed = (week % 52) < closed_weeks_per_year   # % = remainder
            catch = 0.0 if fishing_is_closed else self.harvest_rate * n
            harvest[week] = catch
            stock[week + 1] = max(n + growth - catch, 0.0)   # stock cannot go below 0
        return stock, harvest

    def maximum_sustainable_yield(self) -> float:
        """MSY = r*K/4: the largest catch per week the stock can keep giving forever.

        It happens when the stock is held at half of K and h = r/2.
        """
        return self.growth_rate * self.capacity / 4

    def long_run_stock(self) -> float:
        """Where the stock settles if we keep fishing at rate h: K*(1 - h/r)."""
        return max(self.capacity * (1 - self.harvest_rate / self.growth_rate), 0.0)


class PriceModel:
    """Weekly fish price (UGX per kg) as a random walk that stays within bounds.

    Each week the price moves up or down by a random amount (normal
    distribution with standard deviation `weekly_change_sd`), then it is
    clipped so it never leaves [lower, upper].
    """

    def __init__(self, start_price: float = 12_000, lower: float = 9_000,
                 upper: float = 16_000, weekly_change_sd: float = 400,
                 seed: int = 42) -> None:
        if not lower <= start_price <= upper:
            raise ValueError("start price must be inside the bounds")
        self.start_price = start_price
        self.lower = lower
        self.upper = upper
        self.weekly_change_sd = weekly_change_sd
        self.rng = np.random.default_rng(seed)   # seeded -> repeatable results

    def __repr__(self) -> str:
        return f"PriceModel(start={self.start_price}, bounds=({self.lower}, {self.upper}))"

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Return `weeks` prices. Calling it again gives a NEW path (the
        generator moves on), which is what we want for Monte Carlo."""
        prices = np.zeros(weeks)
        price = self.start_price
        for week in range(weeks):
            prices[week] = price
            price = price + self.rng.normal(0, self.weekly_change_sd)
            price = float(np.clip(price, self.lower, self.upper))  # keep inside bounds
        return prices


def weekly_revenue(harvest_tonnes, price_per_kg) -> np.ndarray:
    """Revenue in UGX = harvest (kg) x price (UGX/kg)."""
    return np.asarray(harvest_tonnes) * KG_PER_TONNE * np.asarray(price_per_kg)


class RiskAssessor:
    """Classifies revenue risk using the Coefficient of Variation (CV).

    CV = standard deviation / mean. It has NO units, so "0.10" means
    "revenue typically moves about 10% away from its average" - whatever the
    currency or the size of the business. That is why it is a fairer rule
    than a raw variance threshold.

    Our rule (a judgement call, stated openly):
        CV < 0.10          -> "Low"     (swings under 10%: easy to plan)
        0.10 <= CV < 0.25  -> "Medium"  (noticeable swings: keep a cash buffer)
        CV >= 0.25         -> "High"    (a quarter or more: hard to plan payroll)
    """

    def __init__(self, low_limit: float = 0.10, high_limit: float = 0.25) -> None:
        if not 0 < low_limit < high_limit:
            raise ValueError("need 0 < low_limit < high_limit")
        self.low_limit = low_limit
        self.high_limit = high_limit

    @staticmethod
    def summary(revenues) -> dict:
        """Mean, median, variance, standard deviation and CV using `statistics`."""
        values = [float(v) for v in revenues]
        if len(values) < 2:
            raise ValueError("need at least 2 values for variance")
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        return {
            "mean": mean,
            "median": statistics.median(values),
            "variance": statistics.variance(values),   # units: UGX squared!
            "std_dev": std_dev,
            "cv": std_dev / mean if mean != 0 else float("inf"),
        }

    def classify(self, revenues) -> str:
        """Return "Low", "Medium" or "High" risk for a revenue series."""
        cv = self.summary(revenues)["cv"]
        if cv < self.low_limit:
            return "Low"
        if cv < self.high_limit:
            return "Medium"
        return "High"

    @staticmethod
    def value_at_risk(annual_revenues, level: float = 0.05) -> dict:
        """5% Value-at-Risk from many simulated annual revenues.

        - revenue_at_5pct: in 5% of simulated years revenue is BELOW this.
        - var_vs_mean: how far below the average that is - the amount of
          revenue "at risk" in a bad (1-in-20) year.
        """
        values = np.asarray(annual_revenues, dtype=float)
        if len(values) == 0:
            raise ValueError("no revenues given")
        cutoff = float(np.percentile(values, level * 100))
        return {"revenue_at_5pct": cutoff, "var_vs_mean": float(values.mean() - cutoff)}


def monte_carlo_annual_revenue(stock: FishStock, n_paths: int = 1000,
                               weeks: int = 52, seed: int = 42) -> np.ndarray:
    """Simulate `n_paths` different years of prices and total the revenue.

    The fish stock (and so the harvest) is the same in every path because the
    stock model has no randomness - only the price changes between paths.
    """
    _, harvest = stock.simulate(weeks)          # "_" = we ignore the stock array
    price_model = PriceModel(seed=seed)
    totals = np.zeros(n_paths)
    for i in range(n_paths):
        prices = price_model.simulate(weeks)
        totals[i] = weekly_revenue(harvest, prices).sum()
    return totals
