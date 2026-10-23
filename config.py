"""Campus layout, congestion levels and classification rules."""
import numpy as np

LEVELS = ["Low", "Medium", "High", "Critical"]
LEVEL_COLORS = {
    "Low": "#2e9e5b",
    "Medium": "#e0b000",
    "High": "#ef7d00",
    "Critical": "#d62839",
}

# capacity = comfortable max number of people, area in m^2,
# x/y = position on the 100x100 campus map used for heatmaps (y=0 is the top / north)
LOCATIONS = {
    "Main Gate":            {"capacity": 150, "area": 400,  "x": 50, "y": 94},
    "Library":              {"capacity": 300, "area": 1200, "x": 20, "y": 25},
    "Cafeteria":            {"capacity": 250, "area": 600,  "x": 55, "y": 50},
    "Admin Block":          {"capacity": 120, "area": 700,  "x": 80, "y": 20},
    "Lecture Hall Complex": {"capacity": 400, "area": 1500, "x": 35, "y": 55},
    "Sports Ground":        {"capacity": 350, "area": 5000, "x": 85, "y": 70},
    "Hostel Entrance":      {"capacity": 180, "area": 500,  "x": 12, "y": 80},
    "Parking Area":         {"capacity": 200, "area": 2500, "x": 75, "y": 92},
}

# occupancy ratio (people / capacity) at which Medium, High, Critical start
DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75)


def classify(count, capacity, thresholds=DEFAULT_THRESHOLDS):
    ratio = count / capacity
    if ratio >= thresholds[2]:
        return "Critical"
    if ratio >= thresholds[1]:
        return "High"
    if ratio >= thresholds[0]:
        return "Medium"
    return "Low"


def level_index(level):
    return LEVELS.index(level)


def level_index_array(counts, capacity, thresholds=DEFAULT_THRESHOLDS):
    """Vectorised classify -> integer 0..3."""
    ratio = np.asarray(counts, dtype=float) / np.asarray(capacity, dtype=float)
    return np.digitize(ratio, thresholds)


def density(count, location):
    """People per square metre."""
    return count / LOCATIONS[location]["area"]
