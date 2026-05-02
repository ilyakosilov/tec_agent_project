"""
Region definitions for TEC analysis.

All agents and tools should use these fixed region IDs instead of inventing
latitude/longitude bounds dynamically. This makes experiments reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """Geographic region used for TEC aggregation."""

    region_id: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str = ""


REGIONS: dict[str, Region] = {
    "equatorial_atlantic": Region(
        region_id="equatorial_atlantic",
        name="Equatorial Atlantic",
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=-60.0,
        lon_max=0.0,
        description="Equatorial region over the Atlantic sector.",
    ),
    "equatorial_africa": Region(
        region_id="equatorial_africa",
        name="Equatorial Africa",
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=0.0,
        lon_max=60.0,
        description="Equatorial region over Africa.",
    ),
    "equatorial_pacific": Region(
        region_id="equatorial_pacific",
        name="Equatorial Pacific",
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=-180.0,
        lon_max=-120.0,
        description="Equatorial region over the Pacific sector.",
    ),
    "midlat_europe": Region(
        region_id="midlat_europe",
        name="Mid-latitude Europe",
        lat_min=35.0,
        lat_max=60.0,
        lon_min=-10.0,
        lon_max=40.0,
        description="European mid-latitude sector.",
    ),
    "midlat_usa": Region(
        region_id="midlat_usa",
        name="Mid-latitude USA",
        lat_min=30.0,
        lat_max=55.0,
        lon_min=-130.0,
        lon_max=-60.0,
        description="North American mid-latitude sector.",
    ),
    "midlat_asia": Region(
        region_id="midlat_asia",
        name="Mid-latitude Asia",
        lat_min=30.0,
        lat_max=55.0,
        lon_min=60.0,
        lon_max=130.0,
        description="Asian mid-latitude sector.",
    ),
    "midlat_south_america": Region(
        region_id="midlat_south_america",
        name="Mid-latitude South America",
        lat_min=-55.0,
        lat_max=-25.0,
        lon_min=-80.0,
        lon_max=-35.0,
        description="South American mid-latitude sector.",
    ),
    "midlat_australia": Region(
        region_id="midlat_australia",
        name="Mid-latitude Australia",
        lat_min=-45.0,
        lat_max=-20.0,
        lon_min=110.0,
        lon_max=155.0,
        description="Australian mid-latitude sector.",
    ),
    "highlat_north": Region(
        region_id="highlat_north",
        name="Northern high latitudes",
        lat_min=65.0,
        lat_max=82.5,
        lon_min=-180.0,
        lon_max=180.0,
        description="Northern high-latitude sector.",
    ),
    "highlat_south": Region(
        region_id="highlat_south",
        name="Southern high latitudes",
        lat_min=-82.5,
        lat_max=-65.0,
        lon_min=-180.0,
        lon_max=180.0,
        description="Southern high-latitude sector.",
    ),
}


def get_region(region_id: str) -> Region:
    """Return a region by ID or raise a clear error."""

    try:
        return REGIONS[region_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(REGIONS))
        raise ValueError(
            f"Unknown region_id: {region_id!r}. Allowed values: {allowed}"
        ) from exc


def list_region_ids() -> list[str]:
    """Return all supported region IDs."""

    return sorted(REGIONS)


def region_to_dict(region: Region) -> dict[str, object]:
    """Convert a Region object to a JSON-serializable dictionary."""

    return {
        "region_id": region.region_id,
        "name": region.name,
        "lat_min": region.lat_min,
        "lat_max": region.lat_max,
        "lon_min": region.lon_min,
        "lon_max": region.lon_max,
        "description": region.description,
    }


def list_regions() -> list[dict[str, object]]:
    """Return all regions as JSON-serializable dictionaries."""

    return [region_to_dict(REGIONS[region_id]) for region_id in list_region_ids()]