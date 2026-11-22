"""Extract plant-month heat input from EIA-923 Schedule 2–5 archives."""

import argparse
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook

from read_eia_activity import HEAT_PREFIXES, MONTHS, NON_COMBUSTION, fuel_group, normalized, value_number


COLUMNS = ["location", "year", "month", "fuel", "prime_mover", "heat_mmbtu"]

def read_fuel_archive(path, year, sites):
    with ZipFile(path) as archive:
        name, = (name for name in archive.namelist() if re.search(r"eia923_schedules_2_3_4_5.*\.xlsx$", name, re.I))
        book = load_workbook(BytesIO(archive.read(name)), read_only=True, data_only=True)
    try:
        sheet, = (sheet for sheet in book if normalized(sheet.title).startswith("page1generationandfuel"))
        rows = iter(sheet.values)
        for row in rows:
            header = [normalized(cell) for cell in row]
            if {"plantid", "reportedprimemover"}.issubset(header):
                break
        else:
            raise ValueError(f"No Generation and Fuel Data header in {path.name}")
        location = header.index("plantid")
        prime_mover = header.index("reportedprimemover")
        fuel = next(header.index(name) for name in ("merfueltypecode", "aerfueltypecode", "reportedfueltypecode")
                    if name in header)
        annual = header.index("totalfuelconsumptionmmbtu")
        monthly = [next(header.index(prefix + month.lower()) for prefix in HEAT_PREFIXES
                        if prefix + month.lower() in header) for month in MONTHS]

        data = []
        invalid = []
        for row in rows:
            plant = row[location]
            code = str(row[fuel] or "").strip().upper()
            if plant not in sites or code in NON_COMBUSTION:
                continue
            heat = [value_number(row[col]) for col in monthly]
            total = value_number(row[annual])
            if min(heat) < 0 or abs(sum(heat) - total) > max(12.0, 0.0001 * total):
                invalid.append((plant, year, code, sum(heat), total, min(heat)))
                continue
            mover = str(row[prime_mover] or "").strip().upper()
            data.extend((plant, year, month, fuel_group(code), mover, value)
                        for month, value in enumerate(heat, 1))
        errors = pd.DataFrame(invalid, columns=["location", "year", "fuel_code", "month_sum",
                                                 "annual_total", "minimum_heat"])
        heat = pd.DataFrame(data, columns=COLUMNS)
        return heat.loc[~heat.location.isin(errors.location)], errors
    finally:
        book.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True, help="Directory containing f923_<year>.zip files")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--ground", type=Path, nargs="+", required=True, help="Facility-hour Parquet files for the plant roster")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sites = set()
    for path in args.ground:
        sites.update(pd.read_parquet(path, columns=["facility_id"]).facility_id.unique())
    args.output.mkdir(parents=True, exist_ok=True)
    for year in sorted(set(args.years)):
        heat, issues = read_fuel_archive(args.archives / f"f923_{year}.zip", year, sites)
        if heat.empty:
            raise ValueError(f"No usable fuel records in {year}")
        heat.to_parquet(args.output / f"plant_heat_{year}.parquet", index=False)
        issues.to_csv(args.output / f"fuel_checks_{year}.csv", index=False)
        print(f"{year}: {heat.location.nunique()} plants; {issues.location.nunique()} excluded by heat checks", flush=True)


if __name__ == "__main__":
    main()
