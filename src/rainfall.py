"""
rainfall.py - Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

Main pieces:
    Region            - 12 monthly rainfall values for one place
    CropRule          - the monthly rainfall range one crop likes
    cosine_similarity - the CORRECT cosine similarity of two vectors
    similarity_matrices() - cosine, Pearson and Euclidean for every pair
    rainy_seasons()   - finds the peaks (rainy seasons) with scipy
"""

# statistics (built in): used for the coefficient of variation.
import statistics

# numpy: arrays and maths.
import numpy as np

# pandas: tables (DataFrames) for the similarity matrices.
import pandas as pd

# scipy.signal.find_peaks finds the "hills" (local highs) in a series.
from scipy.signal import find_peaks

# scipy's cosine DISTANCE (= 1 - cosine similarity), used only to check ours.
from scipy.spatial.distance import cosine as scipy_cosine_distance

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class Region:
    """One region's monthly rainfall.

    Attributes:
        name     - e.g. "Gulu"
        rainfall - NumPy array of 12 values, mm of rain in Jan ... Dec
    """

    def __init__(self, name: str, rainfall) -> None:
        rainfall = np.asarray(rainfall, dtype=float)
        if rainfall.shape != (12,):
            raise ValueError(f"{name}: need exactly 12 monthly values")
        if np.any(rainfall < 0):
            raise ValueError(f"{name}: rainfall cannot be negative")
        self.name = name
        self.rainfall = rainfall

    def __repr__(self) -> str:
        return f"Region('{self.name}', annual={self.annual_total():.0f} mm)"

    def annual_total(self) -> float:
        """Total rain in the year, mm."""
        return float(self.rainfall.sum())

    def mean(self) -> float:
        """Average rain per month, mm."""
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        """Name of the month with the most rain (np.argmax = position of the max)."""
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        """Name of the month with the least rain."""
        return MONTHS[int(np.argmin(self.rainfall))]

    def coefficient_of_variation(self) -> float:
        """Standard deviation / mean. Higher = rain is more uneven through the year."""
        values = list(self.rainfall)
        mean = statistics.mean(values)
        if mean == 0:
            raise ValueError("CV is undefined when mean rainfall is 0")
        return statistics.stdev(values) / mean


class CropRule:
    """The monthly rainfall range (mm) in which a crop does well.

    Below the range -> "Drought risk"; above -> "Waterlogging risk".
    Attributes: crop, min_mm, max_mm, source (where the numbers came from).
    """

    def __init__(self, crop: str, min_mm: float, max_mm: float, source: str = "") -> None:
        if min_mm < 0 or max_mm <= min_mm:
            raise ValueError("need 0 <= min_mm < max_mm")
        self.crop = crop
        self.min_mm = min_mm
        self.max_mm = max_mm
        self.source = source

    def __repr__(self) -> str:
        return f"CropRule('{self.crop}', {self.min_mm}-{self.max_mm} mm/month)"

    def classify(self, rainfall_mm: float) -> str:
        """Label one month's rainfall for this crop."""
        if rainfall_mm < self.min_mm:
            return "Drought risk"
        if rainfall_mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def score(self, rainfall_mm: float) -> int:
        """Same idea as a number, for the heatmap: -1 too dry, 0 good, +1 too wet."""
        if rainfall_mm < self.min_mm:
            return -1
        if rainfall_mm > self.max_mm:
            return 1
        return 0

    def classify_region(self, region: Region) -> list[str]:
        """Labels for all 12 months of a region."""
        return [self.classify(mm) for mm in region.rainfall]


# Crop rules. The FAO gives water needs per GROWING SEASON; we divided by the
# season length to get a monthly range (an approximation - see the notebook).
FAO_SOURCE = "FAO Land & Water, Crop Information pages (fao.org/land-water/databases-and-software/crop-information)"
DEFAULT_CROPS = [
    CropRule("maize", 125, 200, FAO_SOURCE + ": 500-800 mm over a ~4-month season"),
    CropRule("beans", 100, 165, FAO_SOURCE + ": 300-500 mm over a ~3-month season"),
    CropRule("banana", 100, 185, FAO_SOURCE + ": 1,200-2,200 mm per year"),
]


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two vectors = (a . b) / (|a| x |b|).

    a . b  = the dot product: multiply matching values and add them up.
    |a|    = the length (norm) of a: square root of the sum of squares.
    The answer is between -1 and 1 (0 to 1 for rainfall, which is never
    negative). 1 means "the same SHAPE" - it ignores overall size.

    Last year's question used math.cos(), which is the cosine of an ANGLE
    (like cos(60 degrees) = 0.5). That is a different thing entirely.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        raise ValueError("cosine similarity is undefined for an all-zero vector")
    return float(np.dot(a, b) / norm_product)


def cosine_similarity_scipy(a, b) -> float:
    """The same thing via scipy, for checking. scipy returns a DISTANCE, 1 - similarity."""
    return 1 - float(scipy_cosine_distance(a, b))


def similarity_matrices(regions: list[Region]) -> dict[str, pd.DataFrame]:
    """Compare every pair of regions three ways. Returns three tables.

    - cosine:    same shape of rainy/dry months? (ignores how much rain)
    - pearson:   do they go up and down together? (ignores size AND average)
    - euclidean: straight-line distance in mm - big when amounts differ
    """
    names = [r.name for r in regions]
    size = len(regions)
    cosine_m = np.zeros((size, size))
    pearson_m = np.zeros((size, size))
    euclid_m = np.zeros((size, size))
    for i, a in enumerate(regions):          # enumerate gives (position, item)
        for j, b in enumerate(regions):
            cosine_m[i, j] = cosine_similarity(a.rainfall, b.rainfall)
            pearson_m[i, j] = np.corrcoef(a.rainfall, b.rainfall)[0, 1]
            euclid_m[i, j] = np.linalg.norm(a.rainfall - b.rainfall)
    return {
        "cosine": pd.DataFrame(cosine_m, index=names, columns=names),
        "pearson": pd.DataFrame(pearson_m, index=names, columns=names),
        "euclidean": pd.DataFrame(euclid_m, index=names, columns=names),
    }


def rainy_seasons(region: Region, min_prominence: float = 20.0) -> list[str]:
    """Find the rainy-season peaks of a region. Returns month names.

    The year is a circle (December is next to January), but find_peaks only
    sees a straight line, so a peak in Dec or Jan would be missed. Trick: add
    December to the front and January to the end, find peaks, then shift the
    positions back.

    min_prominence: a peak must stand at least this many mm above the dip
    around it, so tiny wiggles are not counted as seasons.
    """
    wrapped = np.concatenate(([region.rainfall[-1]], region.rainfall, [region.rainfall[0]]))
    peak_positions, _ = find_peaks(wrapped, prominence=min_prominence)
    months = [MONTHS[(p - 1) % 12] for p in peak_positions]   # undo the shift
    return months


def load_nasa_power_csv(path) -> pd.DataFrame:
    """Extension: read a NASA POWER monthly rainfall CSV into mm per MONTH.

    NASA POWER (power.larc.nasa.gov) gives PRECTOTCORR = average rain in
    mm per DAY for each month. To get mm per month we multiply by the number
    of days in that month (calendar.monthrange knows about leap years).
    The file starts with a text header; the real table begins at the line
    starting with "PARAMETER". Returns a table: one row per year, Jan..Dec.
    """
    # calendar (built in) - monthrange(year, month) gives (weekday, days_in_month)
    import calendar

    with open(path) as file:
        lines = file.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("PARAMETER"))
    rows = {}
    for line in lines[start + 1:]:
        parts = line.split(",")
        year = int(parts[1])
        mm_per_day = [float(v) for v in parts[2:14]]          # Jan..Dec only
        rows[year] = [mm * calendar.monthrange(year, m + 1)[1]
                      for m, mm in enumerate(mm_per_day)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=MONTHS)


def modality(region: Region, min_prominence: float = 20.0) -> str:
    """ "unimodal" (one rainy season), "bimodal" (two), or a count otherwise."""
    count = len(rainy_seasons(region, min_prominence))
    if count == 1:
        return "unimodal"
    if count == 2:
        return "bimodal"
    return f"{count} peaks"
