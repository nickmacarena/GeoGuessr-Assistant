"""
City/bbox configuration for Mapillary map feature collection.

Each city defines a center point. The collection script will generate
a grid of ~0.01° tiles around each center to query the map features API.
"""

# (city_name, country_code, center_lat, center_lon, grid_radius_deg)
# grid_radius_deg=0.05 means ~10km x 10km area → ~100 tiles of 0.01°
CITIES = [
    # ── Europe ───────────────────────────────────────────────────────
    # Germany
    ("berlin", "DE", 52.52, 13.405, 0.05),
    ("munich", "DE", 48.137, 11.576, 0.05),
    # France
    ("paris", "FR", 48.856, 2.352, 0.05),
    ("lyon", "FR", 45.764, 4.835, 0.05),
    # Netherlands
    ("amsterdam", "NL", 52.370, 4.895, 0.05),
    # Belgium
    ("brussels", "BE", 50.850, 4.350, 0.05),
    # Spain
    ("madrid", "ES", 40.417, -3.704, 0.05),
    ("barcelona", "ES", 41.389, 2.159, 0.05),
    # Italy
    ("rome", "IT", 41.902, 12.496, 0.05),
    ("milan", "IT", 45.464, 9.190, 0.05),
    # Portugal
    ("lisbon", "PT", 38.722, -9.139, 0.05),
    # UK
    ("london", "GB", 51.507, -0.128, 0.05),
    # Sweden
    ("stockholm", "SE", 59.329, 18.069, 0.05),
    # Norway
    ("oslo", "NO", 59.914, 10.752, 0.05),
    # Finland
    ("helsinki", "FI", 60.169, 24.938, 0.05),
    # Poland
    ("warsaw", "PL", 52.230, 21.012, 0.05),
    # Czech Republic
    ("prague", "CZ", 50.076, 14.438, 0.05),
    # Austria
    ("vienna", "AT", 48.208, 16.374, 0.05),
    # Switzerland
    ("zurich", "CH", 47.377, 8.541, 0.05),
    # Denmark
    ("copenhagen", "DK", 55.676, 12.569, 0.05),
    # Greece
    ("athens", "GR", 37.984, 23.728, 0.05),
    # Croatia
    ("zagreb", "HR", 45.815, 15.982, 0.05),

    # ── Americas ─────────────────────────────────────────────────────
    # USA
    ("new_york", "US", 40.713, -74.006, 0.05),
    ("los_angeles", "US", 34.052, -118.244, 0.05),
    ("chicago", "US", 41.878, -87.630, 0.05),
    ("houston", "US", 29.760, -95.370, 0.05),
    # Canada
    ("toronto", "CA", 43.653, -79.383, 0.05),
    ("vancouver", "CA", 49.283, -123.121, 0.05),
    # Mexico
    ("mexico_city", "MX", 19.433, -99.133, 0.05),
    # Brazil
    ("sao_paulo", "BR", -23.551, -46.634, 0.05),
    ("rio", "BR", -22.907, -43.173, 0.05),
    # Argentina
    ("buenos_aires", "AR", -34.604, -58.382, 0.05),
    # Colombia
    ("bogota", "CO", 4.711, -74.072, 0.05),
    # Chile
    ("santiago", "CL", -33.449, -70.669, 0.05),

    # ── Asia ─────────────────────────────────────────────────────────
    # Japan
    ("tokyo", "JP", 35.682, 139.692, 0.05),
    ("osaka", "JP", 34.694, 135.502, 0.05),
    # South Korea
    ("seoul", "KR", 37.567, 126.978, 0.05),
    # Taiwan
    ("taipei", "TW", 25.033, 121.565, 0.05),
    # Thailand
    ("bangkok", "TH", 13.756, 100.502, 0.05),
    # Malaysia
    ("kuala_lumpur", "MY", 3.139, 101.687, 0.05),
    # Indonesia
    ("jakarta", "ID", -6.175, 106.845, 0.05),
    # India
    ("mumbai", "IN", 19.076, 72.878, 0.05),
    ("delhi", "IN", 28.614, 77.209, 0.05),
    # Israel
    ("tel_aviv", "IL", 32.085, 34.782, 0.05),
    # Turkey
    ("istanbul", "TR", 41.009, 28.978, 0.05),

    # ── Oceania ──────────────────────────────────────────────────────
    # Australia
    ("sydney", "AU", -33.869, 151.209, 0.05),
    ("melbourne", "AU", -37.814, 144.963, 0.05),
    # New Zealand
    ("auckland", "NZ", -36.849, 174.763, 0.05),

    # ── Africa ───────────────────────────────────────────────────────
    # South Africa
    ("cape_town", "ZA", -33.925, 18.424, 0.05),
    ("johannesburg", "ZA", -26.205, 28.050, 0.05),
    # Kenya
    ("nairobi", "KE", -1.286, 36.817, 0.05),
    # Nigeria
    ("lagos", "NG", 6.524, 3.379, 0.05),

    # ── Middle East ──────────────────────────────────────────────────
    # UAE
    ("dubai", "AE", 25.205, 55.271, 0.05),
    # Saudi Arabia
    ("riyadh", "SA", 24.713, 46.675, 0.05),

    # ── Eastern Europe / Russia ──────────────────────────────────────
    # Romania
    ("bucharest", "RO", 44.426, 26.102, 0.05),
    # Hungary
    ("budapest", "HU", 47.498, 19.040, 0.05),
]

# Total: 58 cities across 35 countries
