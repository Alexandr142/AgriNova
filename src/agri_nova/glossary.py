"""Plain-language explanations shown in the app's "Learn" tab."""
from __future__ import annotations

GLOSSARY: list[tuple[str, str]] = [
    (
        "Flash drought",
        "A drought that arrives in weeks instead of months: hot, dry, windy weather strips moisture from the soil "
        "faster than usual, often during the crop's most sensitive stages. Northern Plains 2017 is a textbook case.",
    ),
    (
        "Evapotranspiration (ET)",
        "All the water leaving a field as vapour: evaporation from soil plus transpiration through plant leaves. "
        "It is the crop's real water demand. High ET with no rain empties the soil quickly.",
    ),
    (
        "Root-zone soil wetness",
        "How full the soil is where roots reach, from 0 (bone dry) to 1 (saturated). Here it comes from NASA POWER, "
        "a model that blends satellite and weather data - not a probe in your field.",
    ),
    (
        "Soil moisture anomaly",
        "How much wetter or drier the soil is than normal for that time of year. A negative anomaly in July matters "
        "far more than the same number in April.",
    ),
    (
        "SRTM / slope",
        "NASA's radar elevation map. Slope tells us where water runs off and soil can erode: more than ~3% deserves "
        "attention, more than ~8% is high risk.",
    ),
    (
        "Growing degree days (GDD)",
        "Heat accumulated above 10 C. Crops and many pests develop by heat units, not calendar days.",
    ),
    (
        "Heat-stress day / water-stress day",
        "Simple flags used in this tool: a heat-stress day reaches the crop's heat limit (e.g. 35 C for corn); a "
        "water-stress day has dry root-zone soil (wetness < 0.35) and under 10 mm of rain in the last 14 days.",
    ),
    (
        "Cover crop (cereal rye)",
        "A crop grown between cash crops to protect soil, not to harvest. Roots hold soil and open channels for "
        "water; residue shades the surface. Costs: seed, a termination step, and possible spring water/nitrogen use.",
    ),
    (
        "Nitrogen-fixing crops",
        "Legumes such as soybean, alfalfa or field peas host bacteria that turn air nitrogen into plant food, "
        "which can cut purchased fertiliser for the next crop.",
    ),
    (
        "Winter wheat",
        "Seeded in fall, dormant in winter, harvested around July. It grows mostly in cooler, wetter months and "
        "finishes before the hottest part of summer, and its deep roots and different pests break corn/soy cycles.",
    ),
    (
        "Z-score",
        "How many 'typical swings' a season is above or below the 2012-2022 average. +2 means unusually high; "
        "-2 unusually low. With only 11 seasons, treat it as a guide.",
    ),
    (
        "Confidence interval (Wilson)",
        "A range showing how sure we can be about 'this rotation was supported in X of N seasons'. With few seasons "
        "the range is wide - that is honest uncertainty, not an error.",
    ),
    (
        "Frost risk / frost-free days",
        "A frost day is a day when the air drops to 0 C or below. The days between the last spring frost and the first "
        "fall frost are the frost-free window: it limits which crops finish, and how long a fall cover crop or winter wheat "
        "has to establish. Gridded weather data runs warmer than a frost-prone low spot, so your own field can frost earlier or later.",
    ),
    (
        "Grid cell (regional map)",
        "NASA POWER weather comes from the MERRA-2 model, which divides the planet into cells about 0.5 x 0.625 degrees "
        "(roughly 50 km). Everything inside a cell shares one value, so the regional map compares neighbouring cells, "
        "not individual fields.",
    ),
]

SATELLITE_VS_GROUND = (
    "**Satellites and models** see every field the same way, every day, back decades - but they measure from "
    "far away: NASA POWER cells are about 50 km wide, and gridded values describe surfaces and model estimates, "
    "not what is happening at root depth in your field.\n\n"
    "**Ground measurements** (rain gauges, soil probes, soil surveys, your own yield records) are precise for one "
    "spot but miss everything around them.\n\n"
    "**Combining both** is more reliable than either: NASA data tells you *when and how widely* something happened; "
    "USDA soil survey, crop records and your own fields tell you *how it played out here*. That is why this tool "
    "combines NASA data with USDA soil survey and crop records, and why local knowledge is the final check."
)

LIMITS: list[str] = [
    "Scores and signals are transparent rules of thumb built from general agronomy - not a yield forecast and not "
    "agronomic advice.",
    "Soil wetness is modeled (NASA POWER / MERRA-2), not measured by a satellite or probe, and it describes a "
    "grid cell, not your field.",
    "Eleven seasons (2012-2022) is a short record. Percentages across seasons carry wide uncertainty.",
    "Scenario sliders apply a simple assumed sensitivity of soil wetness to rain and heat; they are not climate projections.",
    "Local seeding dates, varieties, winter survival, prices, equipment and insurance rules are not modeled.",
    "The regional map compares ~50 km weather-model cells; it cannot show field-to-field differences inside one cell.",
    "Irrigation is only used to soften the water-saving scores; water rights, pumping capacity and cost are not modeled.",
]

HUMAN_JUDGMENT: list[str] = [
    "Whether a cover crop can be seeded and terminated on time on your fields and soils.",
    "Whether winter wheat survives your winters and has a market you can reach.",
    "Your yield history, field quirks (wet spots, sandy knolls) and what you know about this year's weather.",
    "Crop-insurance, program and rental-agreement rules that affect rotation choices.",
    "Talk to your county extension agent, crop consultant or NRCS office before changing a rotation.",
]
