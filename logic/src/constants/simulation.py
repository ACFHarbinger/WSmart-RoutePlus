"""
Simulation and Physics constants.

This module defines physical constants, problem constraints, and metric definitions
for the WSmart Route+ simulation engine. Used by:
- logic/src/pipeline/simulations/manager.py (multi-day simulation orchestrator)
- logic/src/envs/*.py (problem environment physics)
- logic/src/data/generators/*.py (instance generation)

Metric Naming Conventions
--------------------------
Three metric lists serve different scopes:

- **DAY_METRICS**: Single-day statistics (includes "day" and "tour" fields)
  Used in: Daily log files, per-day CSV output
  Example: {"day": 5, "kg": 125.3, "km": 45.2, "tour": [0, 3, 7, 0]}

- **METRICS**: Core performance metrics (overflows, kg, km, reward, profit, time)
  Used in: Summary tables, per-policy comparisons, HPO objectives
  Example: {"kg": 5234.1, "km": 1230.5, "reward": 615.25, "profit": 2617.05, "time": 3.42,}

- **SIM_METRICS**: Overall simulation statistics (adds "days")
  Used in: Final summary reports, simulation metadata
  Example: {"days": 31 "kg": 5234.1, ...}

- **LOSS_KEYS**: Neural network training metrics
  Used in: Training logs, TensorBoard/WandB logging
  Example: {"nll": 0.523, "reinforce_loss": 1.234, "baseline_loss": 0.711}

Physical Constants
------------------
Distance calculations use geodesic formulas (Haversine, Vincenty) for real-world
accuracy. Two Earth radius values support different precision requirements:
- EARTH_RADIUS: Simplified sphere model (faster, ±0.5% error)
- EARTH_WMP_RADIUS: WGS84 ellipsoid model (higher precision for GPS data)

Attributes:
  EARTH_RADIUS: Mean Earth radius (kilometers) - Spherical approximation
  EARTH_WMP_RADIUS: WGS84 equatorial radius (meters) - Ellipsoid model
  METRICS: Core performance metrics (overflows, kg, km, reward, profit, time)
  SIM_METRICS: Overall simulation statistics (adds "days")
  DAY_METRICS: Single-day statistics (includes "day" and "tour" fields)
  LOSS_KEYS: Neural network training metrics

Example:
  >>> from logic.src.constants import EARTH_RADIUS, EARTH_WMP_RADIUS, METRICS, SIM_METRICS, DAY_METRICS, LOSS_KEYS
  >>> EARTH_RADIUS
  6371
"""

from typing import List

# Distance Matrix - Earth Radius Models
# --------------------------------------
# Used for geodesic distance calculation from GPS coordinates.
# Formula: Haversine distance = 2 * R * arcsin(√(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))

# Mean Earth radius (kilometers) - Spherical approximation
# Used in: Haversine distance formula for simplified calculations
# Accuracy: ±0.5% error globally (max error at poles)
# Use when: Speed > precision (data generation, quick prototypes)
EARTH_RADIUS: int = 6371  # km (mean radius)

# WGS84 equatorial radius (meters) - Ellipsoid model
# Used in: Vincenty formula for high-precision GPS distance calculations
# Accuracy: Sub-millimeter precision (geodetic standard)
# Use when: Real-world GPS data (OpenStreetMap, Google Maps integration)
EARTH_WMP_RADIUS: int = 6378137  # meters (equatorial radius, WGS84 datum)

# WSmart Route+ Simulation Metrics
# ---------------------------------
# Performance indicators for waste collection optimization.

# Core performance metrics (reported daily and overall)
# Unit specifications:
# - overflows: count (integer, bins that exceeded capacity)
# - kg: kilograms (float, total waste collected)
# - ncol: count (integer, number of collections performed)
# - kg_lost: kilograms (float, waste lost due to overflows)
# - km: kilometers (float, total distance traveled)
# - kg/km: kg per km (float, collection efficiency ratio)
# - profit: currency units (float, revenue - cost). Day logs do not store reward.
METRICS: List[str] = [
    "overflows",  # Bin overflows (minimize to 0). Not a route-feasibility violation.
    "kg",  # Waste collected (maximize)
    "ncol",  # Collection count (minimize for efficiency)
    "kg_lost",  # Waste lost above bin capacity (minimize to 0)
    "km",  # Distance traveled (minimize)
    "kg/km",  # Ratio of summed kg to summed km
    "profit",  # Net profit in EUR (maximize, primary objective)
    "time",  # Policy time in s: packing + routing + route improvement (minimize)
    "day_wall_time",  # Whole run_day wall time in s (minimize)
    "days_simulated",  # Days the sample attempted
    "collection_days",  # Days with a non-empty executed tour
]

# Sample columns. ``time_spent`` is CMPTPTC-only and is not in this default list.
SIM_METRICS: List[str] = list(METRICS)

# Daily log metrics (add mandatory nodes selected before tour)
# Additional fields:
# - mandatory_nodes: List[int] (bin IDs selected as mandatory before routing)
# - tour: List[int] (node sequence, e.g., [0, 5, 12, 8, 0] for depot→5→12→8→depot)
# - solver_status: str (backend status for this day, kept when the tour is empty)
DAY_METRICS: List[str] = METRICS + ["mandatory_nodes", "tour", "solver_status"]

# Neural network training loss components
# Used in: RL training loops, logged to WandB/TensorBoard
# - nll: negative log-likelihood of actions (policy gradient loss)
# - reinforce_loss: REINFORCE policy gradient loss (with baseline)
# - baseline_loss: value network MSE loss (for critic training)
LOSS_KEYS: List[str] = ["nll", "reinforce_loss", "baseline_loss"]

# Problem Definition - Constraint Parameters
# --------------------------------------------
# Physical and operational constraints for VRP variants.

# Maximum bin fill level (normalized)
# Range: [0.0, 100.0] where 100.0 = full capacity
# Overflow occurs when fill > MAX_CAPACITY_PERCENT after waste generation
# Used in: Bin state validation, overflow counting, reward penalty
MAX_CAPACITY_PERCENT = 100.0  # percent (0-100 range)

# Maximum route length constraints by problem size
# Maps number of customer locations → max route length (hops, excluding depot returns)
# Prevents unbounded route lengths in waste-collecting and selective problems.

# Default vehicle capacity (kilograms)
# Used in: Capacitated VRP variants (CVRP, CMPTP, CMPTPTC)
# Route terminates when cumulative collected waste ≥ VEHICLE_CAPACITY
# Typical real-world values: 80-120 kg for small trucks, 200-300 kg for large trucks
VEHICLE_CAPACITY: float = 200.0  # kg (default for synthetic instances)

# Supported Problem Types
# -----------------------
# Environment registry for problem selection via CLI/config.
# Format: {problem_name} → logic/src/envs/{problem_name}.py
PROBLEMS: List[str] = [
    "ptp",  # Profitable Tour Problem (maximize profit - cost)
    "cmptp",  # Capacitated Multi-vehicle Profitable Tour Problem (adds the vehicle capacity constraint)
    "cmptptc",  # Capacitated Multi-vehicle Profitable Tour Problem with Time Constraints (capacity + shift time budget)
]
