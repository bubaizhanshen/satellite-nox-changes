"""Extract monthly plant fuel heat input from the EIA-923 source workbooks."""

import argparse
import calendar
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook


MONTHS = list(calendar.month_name)[1:]
HEAT_PREFIXES = ("totmmbtu", "mmbtutot", "totalfuelconsumptionmmbtu")
NON_COMBUSTION = {"NUC", "SUN", "WND", "HYC", "HYD", "GEO", "HPS", "WAT", "PUR", "WH"}
COAL = {"COL", "ANT", "BIT", "LIG", "SUB", "WC", "RC"}
NATURAL_GAS = {"NG"}


def normalized(item):
    return re.sub(r"[^a-z0-9]", "", str(item).lower())


def value_number(value):
    if value in (None, "", "."):
        return 0.0
    return float(value)


def fuel_group(code):
    if code in COAL:
        return "coal"
    if code in NATURAL_GAS:
        return "gas"
    return "other"


def read_year(path, year, plants):
    with ZipFile(path) as archive:
        name, = [item for item in archive.namelist()
                 if re.search(r"eia923_schedules_2_3_4_5.*\.xlsx$", item, flags=re.I)]
        workbook = load_workbook(BytesIO(archive.read(name)), data_only=True, read_only=True)
    try:
        sheet, = [item for item in workbook.sheetnames if "page1generationandfuel" in normalized(item)]
        rows = workbook[sheet].values
        for row in rows:
            header = [normalized(cell) for cell in row]
            if "plantid" in header:
                break
        else:
            raise ValueError(f"No plant identifier in {path.name} {sheet}")

        plant_col = header.index("plantid")
        fuel_name = next(name for name in ("merfueltypecode", "aerfueltypecode", "reportedfueltypecode")
                         if name in header)
        fuel_col = header.index(fuel_name)
        frequency_name = next(name for name in ("respondentfrequency", "reportingfrequency") if name in header)
        report_col = header.index(frequency_name)
        annual_col = header.index("totalfuelconsumptionmmbtu")
        columns = {}
        for month, name in enumerate(MONTHS, 1):
            matches = [i for i, title in enumerate(header)
                       if title in [prefix + name.lower() for prefix in HEAT_PREFIXES]]
            if len(matches) != 1:
                raise ValueError(f"Fuel heat column for {name}: {matches}, header: {header}")
            columns[month] = matches[0]

        output = []
        mismatches = []
        for row in rows:
            plant = row[plant_col]
            if plant not in plants:
                continue
            code = str(row[fuel_col] or "").strip().upper()
            if code in NON_COMBUSTION:
                continue
            fuel = fuel_group(code)
            report = str(row[report_col]).strip().upper()
            annual_total = value_number(row[annual_col])
            month_total = sum(value_number(row[col]) for col in columns.values())
            if abs(month_total - annual_total) > max(12.0, 0.0001 * annual_total):
                mismatches.append((plant, code, annual_total, month_total))
            for month, col in columns.items():
                raw = row[col]
                output.append((plant, year, month, code, fuel, report, raw in (None, "", "."),
                               value_number(raw)))
    finally:
        workbook.close()

    if mismatches:
        raise ValueError(f"Monthly heat input does not sum to the published annual total: {mismatches[:5]}")
    data = pd.DataFrame(output, columns=["location", "year", "month", "code", "fuel", "frequency",
                                         "missing_cell", "heat_mmbtu"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--activity", type=Path, default=Path("eia923_monthly_activity.csv"))
    args = parser.parse_args()
    sites = set(pd.read_csv(args.panel, usecols=["location"]).location)
    data = pd.concat([read_year(Path(f"f923_{year}.zip"), year, sites) for year in args.years])
    if data.empty or data.heat_mmbtu.lt(0).any():
        raise ValueError("Missing data or negative fuel heat input")
    columns = ["location", "year", "month"]
    values = data.pivot_table(index=columns, columns="fuel", values="heat_mmbtu", aggfunc="sum",
                              fill_value=0.0).add_prefix("heat_")
    diagnostics = data.groupby(columns).agg(source_rows=("heat_mmbtu", "size"),
                                            cells_blank=("missing_cell", "sum"),
                                            monthly_report=("frequency", lambda x: x.eq("M").all()))
    monthly = values.join(diagnostics).reset_index().sort_values(columns)
    monthly.to_csv(args.activity, index=False)
    print(monthly.groupby("year").agg(plants=("location", "nunique"), months=("month", "count"),
                                      all_monthly_report=("monthly_report", "mean")).to_string())
    print("Fuel categories and reporter frequencies:")
    print(data.groupby(["code", "fuel", "frequency"], dropna=False).agg(
        rows=("code", "size"), blank=("missing_cell", "sum"), heat=("heat_mmbtu", "sum"),
    ).to_string())


if __name__ == "__main__":
    main()
