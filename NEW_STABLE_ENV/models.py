# models.py
# =============================================================================
# Typed Pydantic models for PrecisionIrrigation-v0 OpenEnv environment.
# Represents a Punjab rice-paddy irrigation simulation.
# =============================================================================

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from openenv.core.env_server import Action, Observation, State


# -- Action --------------------------------------------------------------------

class IrrigationAction(Action):
    """
    Discrete action for the irrigation agent.

    Action space: Discrete(5)
        0 = no_water    - No irrigation today.
        1 = low         - Low irrigation (10% pump capacity, 50 L).
        2 = medium      - Medium irrigation (30% pump capacity, 150 L).
        3 = high        - High irrigation (50% pump capacity, 250 L).
        4 = drain       - Drain excess water (lowers soil moisture by 0.15).
    """
    action: int = Field(
        ...,
        ge=0,
        le=4,
        description=(
            "0=no_water | 1=low_50L | 2=medium_150L | 3=high_250L | 4=drain"
        ),
    )


# -- Observation sub-models ----------------------------------------------------

class DayForecast(BaseModel):
    """Single day weather forecast from mock IMD API."""
    day_offset: int              # +1, +2, or +3 days ahead
    rain_mm: float               # predicted rainfall in mm
    temp_celsius: float          # predicted temperature degC
    humidity_pct: float          # relative humidity (0-100)
    power_cut_risk: float        # probability of power cut today (0-1)


class IrrigationObservation(Observation):
    """
    Full observation returned from the Punjab irrigation environment.

    Dict-structured for Gymnasium Dict observation space compatibility.
    """
    episode_id: Optional[str] = None

    # -- Core agronomic state -------------------------------------
    day: int                         # current simulation day (1-indexed)
    total_days: int                  # episode length
    soil_moisture: float             # 0.0-1.0 (fraction of field capacity)
    crop_growth: float               # 0.0-1.0 (yield potential realized)
    water_reserve: float             # litres remaining in tube-well tank
    power_status: int                # 1 = grid ON, 0 = power CUT

    # -- Weather --------------------------------------------------
    current_rain_mm: float
    current_temp_celsius: float
    weather_forecast: List[DayForecast]  # next 3 days
    season: str                          # pre_monsoon/monsoon/post_monsoon/rabi

    # -- Episode tracking -----------------------------------------
    cumulative_water_used: float     # total litres irrigated so far
    stress_days: int                 # days where crop experienced moisture stress
    last_action_result: str          # human-readable outcome of last action
    water_efficiency_index: float    # yield/water ratio vs baseline (>1.2 = good)

    # -- LLM agent context ----------------------------------------
    echoed_message: str              # full natural-language state description


# -- State (internal metadata) -------------------------------------------------

class IrrigationState(State):
    """Internal episode metadata, returned by /state endpoint."""
    task_name: str = ""
    day: int = 0
    total_days: int = 30
    seed: int = 42
    scenario: str = "monsoon"        # monsoon/pre_monsoon/drought
    cumulative_reward: float = 0.0
    crop_growth: float = 0.0
    soil_moisture: float = 0.0
    water_reserve: float = 0.0
    cumulative_water_used: float = 0.0
