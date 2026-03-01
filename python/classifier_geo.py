"""Geographic classifier using EXIF GPS data.

Replaces src/classifier_geo.js.

No ML — reads EXIF GPS tags and performs reverse geocoding.
Output: ["Country1", "Country2"] or []

Reference: src/classifier_geo.js
"""
import json
import os
import sys

import exifread

# reverse_geocoder prints to stdout on first load — redirect to stderr
import io
_orig_stdout = sys.stdout
sys.stdout = sys.stderr
import reverse_geocoder as rg
sys.stdout = _orig_stdout

import base_classifier

# ISO 3166-1 alpha-2 → country name (matches JS geo-reverse output)
# Generated from reverse_geocoder's own dataset
CC_TO_NAME = {}
try:
    # Build lookup from reverse_geocoder's internal data at import time
    import csv as _csv
    _rg_data = os.path.join(os.path.dirname(rg.__file__), 'rg_cities1000.csv')
    if os.path.isfile(_rg_data):
        # reverse_geocoder doesn't expose country names, so we use pycountry if available
        pass
except Exception:
    pass

# Fallback: use pycountry for authoritative country names
try:
    import pycountry
    CC_TO_NAME = {c.alpha_2: c.name for c in pycountry.countries}
except ImportError:
    # Minimal fallback with common codes
    CC_TO_NAME = {
        'US': 'United States', 'GB': 'United Kingdom', 'FR': 'France',
        'DE': 'Germany', 'IT': 'Italy', 'ES': 'Spain', 'JP': 'Japan',
        'CN': 'China', 'IN': 'India', 'BR': 'Brazil', 'CA': 'Canada',
        'AU': 'Australia', 'RU': 'Russia', 'KR': 'South Korea',
        'MX': 'Mexico', 'ID': 'Indonesia', 'TR': 'Turkey', 'SA': 'Saudi Arabia',
        'AR': 'Argentina', 'ZA': 'South Africa', 'TH': 'Thailand',
        'EG': 'Egypt', 'NL': 'Netherlands', 'SE': 'Sweden', 'NO': 'Norway',
        'PL': 'Poland', 'BE': 'Belgium', 'CH': 'Switzerland', 'AT': 'Austria',
        'PT': 'Portugal', 'GR': 'Greece', 'CZ': 'Czech Republic',
        'NZ': 'New Zealand', 'IE': 'Ireland', 'DK': 'Denmark', 'FI': 'Finland',
    }


def convert_dms_to_dd(dms_values, direction):
    """Convert degrees/minutes/seconds to decimal degrees.

    Matches JS ConvertDMSToDD function.
    """
    degrees = float(dms_values[0].num) / float(dms_values[0].den)
    minutes = float(dms_values[1].num) / float(dms_values[1].den)
    seconds = float(dms_values[2].num) / float(dms_values[2].den)

    dd = degrees + minutes / 60.0 + seconds / 3600.0

    if direction in ('S', 'W'):
        dd = -dd

    return dd


def main():
    paths = base_classifier.get_paths()

    for path in paths:
        try:
            with open(path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            lat_tag = tags.get('GPS GPSLatitude')
            lat_ref = tags.get('GPS GPSLatitudeRef')
            lon_tag = tags.get('GPS GPSLongitude')
            lon_ref = tags.get('GPS GPSLongitudeRef')

            if not all([lat_tag, lat_ref, lon_tag, lon_ref]):
                base_classifier.output_result([])
                continue

            lat = convert_dms_to_dd(lat_tag.values, str(lat_ref))
            lon = convert_dms_to_dd(lon_tag.values, str(lon_ref))

            # reverse_geocoder returns dicts with 'cc' (country code)
            # The JS version outputs country names, so we convert cc → name
            results = rg.search([(lat, lon)])
            if results:
                countries = []
                for r in results:
                    cc = r.get('cc', '')
                    if cc and cc in CC_TO_NAME:
                        countries.append(CC_TO_NAME[cc])
                    elif cc:
                        countries.append(cc)

                # Deduplicate
                seen = set()
                unique = []
                for c in countries:
                    if c not in seen:
                        seen.add(c)
                        unique.append(c)
                base_classifier.output_result(unique)
            else:
                base_classifier.output_result([])

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
