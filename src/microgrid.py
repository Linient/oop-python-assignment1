"""
microgrid.py - Mini-Project 2: Solar Micro-Grid Dispatch Planner.

The health centre must meet two demands every day:
    3x + 2y = D1   (daytime load, kWh)
    4x + 1y = D2   (critical-equipment load, kWh)
where x = energy from solar panels and y = energy from batteries (kWh).

In matrix form that is  A @ [x, y] = [D1, D2]  with
    A = [[3, 2],
         [4, 1]]
Solving the system tells us how much of each source to use.

Main pieces:
    MicroGrid           - holds A, checks it, solves days, computes cost
    HybridMicroGrid     - extension: adds a diesel generator (3 x 3 system)
    read_positive_number() - asks the user for a number until it is valid
    generate_demand_csv()  - makes 30 days of fake-but-realistic demand
    load_demand_csv()      - reads that file back
"""

# csv is built into Python: reads and writes comma-separated text files.
import csv

# pathlib.Path represents a file path in a way that works on Windows and Mac.
from pathlib import Path

# numpy: arrays and maths (see metrics.py).
import numpy as np

# scipy is a science library built on numpy.
#   scipy.linalg.solve  - solves a linear system A @ x = b
#   scipy.optimize.nnls - "non-negative least squares": finds the closest
#                         answer where every value is >= 0
from scipy.linalg import solve
from scipy.optimize import nnls

# Prices from the assignment (illustrative), in Uganda shillings per kWh.
SOLAR_PRICE_UGX = 150
BATTERY_PRICE_UGX = 450


class MicroGrid:
    """A micro-grid described by its coefficient matrix A.

    Attributes:
        coefficients - the matrix A (NumPy 2-D array)
        source_names - names of the energy sources, one per column of A
    """

    def __init__(self, coefficients=None, source_names=("solar", "battery")) -> None:
        if coefficients is None:                 # use the assignment's system
            coefficients = [[3, 2],
                            [4, 1]]
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.source_names = list(source_names)

        rows, columns = self.coefficients.shape  # .shape gives (rows, columns)
        if rows != columns:
            raise ValueError("the coefficient matrix must be square")
        if columns != len(self.source_names):
            raise ValueError("need one source name per column")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sources={self.source_names})"

    # --- Is the system well-posed? ----------------------------------------

    def determinant(self) -> float:
        """The determinant of A.

        If it is 0, the equations are not independent and there is no single
        answer (either none or infinitely many). Non-zero means one answer.
        """
        return float(np.linalg.det(self.coefficients))

    def condition_number(self) -> float:
        """The condition number of A (np.linalg.cond).

        It measures how much small errors in the demand get magnified in the
        answer. Near 1 = very stable. In the thousands or more = a tiny
        measuring error in D1 or D2 can swing x and y a lot.
        """
        return float(np.linalg.cond(self.coefficients))

    def is_well_posed(self, max_condition: float = 1e6) -> bool:
        """True if the system has one stable answer."""
        return abs(self.determinant()) > 1e-12 and self.condition_number() < max_condition

    # --- Solving ----------------------------------------------------------

    def solve_demands(self, demands) -> np.ndarray:
        """Solve A @ usage = demands for one day. Returns an array of usage."""
        demands = np.asarray(demands, dtype=float)
        if demands.shape != (len(self.source_names),):
            raise ValueError(f"need exactly {len(self.source_names)} demand values")
        if np.any(demands < 0):
            raise ValueError("demand cannot be negative")
        if not self.is_well_posed():
            raise np.linalg.LinAlgError("system is singular or badly conditioned")
        return solve(self.coefficients, demands)

    def solve_day(self, d1: float, d2: float) -> tuple[float, float]:
        """Solve one day and return (solar_kwh, battery_kwh)."""
        x, y = self.solve_demands([d1, d2])
        return float(x), float(y)

    def solve_days_loop(self, demand_matrix) -> np.ndarray:
        """Solve many days one at a time with a normal Python for-loop.

        demand_matrix has shape (number_of_sources, number_of_days),
        e.g. 2 rows x 30 columns. Column j is day j. Returns the same shape.
        """
        demand_matrix = np.asarray(demand_matrix, dtype=float)
        results = np.zeros_like(demand_matrix)        # empty answer grid
        for day in range(demand_matrix.shape[1]):     # loop over the columns
            results[:, day] = solve(self.coefficients, demand_matrix[:, day])
        return results

    def solve_days_vectorised(self, demand_matrix) -> np.ndarray:
        """Solve ALL days in a single call.

        scipy.linalg.solve accepts a right-hand side with many columns and
        solves them together, which avoids Python's slow loop.
        """
        demand_matrix = np.asarray(demand_matrix, dtype=float)
        return solve(self.coefficients, demand_matrix)

    # --- Infeasible days ---------------------------------------------------

    @staticmethod
    def infeasible_days(usage_matrix) -> np.ndarray:
        """True for each day (column) where any source is negative.

        Negative energy is physically impossible - a panel cannot "un-produce".
        @staticmethod means this method does not need `self`.
        """
        usage_matrix = np.asarray(usage_matrix, dtype=float)
        return np.any(usage_matrix < 0, axis=0)   # axis=0: check down each column

    def solve_day_non_negative(self, demands) -> tuple[np.ndarray, float]:
        """Our handling strategy for infeasible days: NNLS.

        nnls finds the usage (all values >= 0) that gets as close as possible
        to the demands. It returns (usage, residual). The residual is how far
        we miss the demand, in kWh - 0 means we met it exactly.
        """
        usage, residual = nnls(self.coefficients, np.asarray(demands, dtype=float))
        return usage, float(residual)

    def solve_days_safely(self, demand_matrix) -> tuple[np.ndarray, list]:
        """Solve every day; re-solve infeasible days with NNLS.

        Returns (usage_matrix, report) where report is a list of dictionaries
        describing each day that had to be fixed.
        """
        usage = self.solve_days_vectorised(demand_matrix)
        report = []
        for day in np.where(self.infeasible_days(usage))[0]:   # indices of bad days
            original = usage[:, day].copy()
            fixed, shortfall = self.solve_day_non_negative(np.asarray(demand_matrix)[:, day])
            usage[:, day] = fixed
            report.append({"day": int(day) + 1, "original": original,
                           "fixed": fixed, "shortfall_kwh": shortfall})
        return usage, report

    # --- Cost -------------------------------------------------------------

    @staticmethod
    def daily_cost(solar_kwh, battery_kwh,
                   solar_price: float = SOLAR_PRICE_UGX,
                   battery_price: float = BATTERY_PRICE_UGX):
        """Energy cost in UGX: kWh used x price per kWh, added for both sources.

        Works for single numbers or whole arrays of days.
        """
        return np.asarray(solar_kwh) * solar_price + np.asarray(battery_kwh) * battery_price


class HybridMicroGrid(MicroGrid):
    """Extension: a micro-grid with a diesel generator (z) as a third source.

    It INHERITS everything from MicroGrid (checks, solving, NNLS) and only
    changes the default matrix. Our third constraint (our own choice):
        1x + 1y + 1z = D3   (total energy the centre uses in a day)
    """

    def __init__(self, coefficients=None) -> None:
        if coefficients is None:
            coefficients = [[3, 2, 0],
                            [4, 1, 0],
                            [1, 1, 1]]
        # super() calls the PARENT class's __init__ so we reuse its checks.
        super().__init__(coefficients, source_names=("solar", "battery", "diesel"))

    def solve_day(self, d1: float, d2: float, d3: float) -> tuple[float, float, float]:
        """Solve one day with three demands. Returns (solar, battery, diesel)."""
        x, y, z = self.solve_demands([d1, d2, d3])
        return float(x), float(y), float(z)


# ===========================================================================
# Input helpers
# ===========================================================================

def read_positive_number(prompt: str, input_function=input) -> float:
    """Keep asking until the user types a valid number that is >= 0.

    Rejects: empty input, text that is not a number, negative numbers.

    `input_function` is normally Python's built-in input(). Tests can pass a
    fake one instead, so we can test this without a person typing.
    """
    while True:                                   # loop until we `return`
        text = input_function(prompt).strip()     # .strip() removes spaces
        if text == "":
            print("  Please type a value - it cannot be empty.")
            continue                              # go back to the top of the loop
        try:
            value = float(text)                   # may fail, e.g. for "abc"
        except ValueError:
            print(f"  '{text}' is not a number. Try again.")
            continue
        if value < 0:
            print("  Demand cannot be negative. Try again.")
            continue
        return value


def generate_demand_csv(path, days: int = 30, seed: int = 42) -> Path:
    """Create a CSV of daily demands with a weekly pattern plus random noise.

    - D1 (daytime load) is higher on weekdays (clinics, lights, fans) and
      lower at weekends, around 120 kWh.
    - D2 (critical equipment: fridges for vaccines, oxygen) is steadier,
      around 135 kWh, with its own noise. On a few days it comes
      close to D1, which (as we will see) makes the system infeasible.
    The seed makes the "random" numbers identical every time we run.
    """
    rng = np.random.default_rng(seed)
    day_numbers = np.arange(days)                          # 0, 1, ..., 29
    weekly_wave = np.sin(2 * np.pi * day_numbers / 7)      # repeats every 7 days
    d1 = 120 + 15 * weekly_wave + rng.normal(0, 8, days)   # normal(mean, sd, how many)
    d2 = 135 + 5 * weekly_wave + rng.normal(0, 15, days)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)        # create the folder if missing
    with open(path, "w", newline="") as file:              # "w" = write mode
        writer = csv.writer(file)
        writer.writerow(["day", "d1_kwh", "d2_kwh"])        # header row
        for day, a, b in zip(day_numbers + 1, d1, d2):     # zip walks lists together
            writer.writerow([day, round(a, 2), round(b, 2)])
    return path


def load_demand_csv(path) -> np.ndarray:
    """Read the demand CSV. Returns a 2 x N array: row 0 = D1, row 1 = D2."""
    d1, d2 = [], []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):   # each row becomes {"day": .., "d1_kwh": ..}
            d1.append(float(row["d1_kwh"]))
            d2.append(float(row["d2_kwh"]))
    if not d1:
        raise ValueError("the CSV file has no data rows")
    return np.array([d1, d2])
