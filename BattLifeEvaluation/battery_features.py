
"""
Battery Feature Extractor
-------------------------
Generates:
  1) DOD histogram (Depth of Discharge, %SOC swings)
  2) Throughput curves (cumulative Ah / Wh)
  3) ΔV trend (max_cell_voltage - min_cell_voltage)
  4) ΔT trend (max_cell_temperature - min_cell_temperature)
  5) Basic cycle table (cycles with >= min_dod % SOC swing)

Usage:
  python battery_features.py --csv /path/to/wf512.csv --outdir ./out

Notes:
  - Plots use matplotlib with separate figures and default colors (no seaborn, no subplots).
  - Cycle detection uses turning-point method on smoothed SOC; for production, consider rainflow counting.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns:
        raise ValueError("CSV must contain a 'time' column (UNIX seconds).")
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def add_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df['delta_t_s'] = df['datetime'].diff().dt.total_seconds()
    # invalidate non-positive or null intervals
    df.loc[(df['delta_t_s'] is None) | (df['delta_t_s'] <= 0), 'delta_t_s'] = np.nan
    df['delta_t_h'] = df['delta_t_s'] / 3600.0
    return df

def add_power_and_throughput(df: pd.DataFrame) -> pd.DataFrame:
    if not {'battery_voltage','battery_current'}.issubset(df.columns):
        raise ValueError("CSV must contain 'battery_voltage' and 'battery_current'.")
    df['power_kW'] = (df['battery_voltage'] * df['battery_current']) / 1000.0

    # Ah / Wh integrals (absolute and signed components)
    df['abs_Ah'] = df['battery_current'].abs() * df['delta_t_h']
    df['chg_Ah'] = (-df['battery_current'].clip(upper=0)) * df['delta_t_h']  # charging magnitude
    df['dch_Ah'] = (df['battery_current'].clip(lower=0)) * df['delta_t_h']   # discharging magnitude

    df['abs_Wh'] = df['power_kW'].abs() * 1000.0 * df['delta_t_h']  # Wh
    df['chg_Wh'] = (-df['power_kW'].clip(upper=0)) * 1000.0 * df['delta_t_h']
    df['dch_Wh'] = (df['power_kW'].clip(lower=0)) * 1000.0 * df['delta_t_h']

    df['cum_Ah'] = df['abs_Ah'].cumsum()
    df['cum_Wh'] = df['abs_Wh'].cumsum()
    return df

def add_deltaV_deltaT(df: pd.DataFrame, roll_window:int=30) -> pd.DataFrame:
    required_v = {'max_cell_voltage','min_cell_voltage'}
    required_t = {'max_cell_temperature','min_cell_temperature'}
    if not required_v.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_v}")
    if not required_t.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_t}")

    df['deltaV'] = df['max_cell_voltage'] - df['min_cell_voltage']
    df['deltaT'] = df['max_cell_temperature'] - df['min_cell_temperature']

    # Rolling median for trend
    df['deltaV_roll'] = df['deltaV'].rolling(window=roll_window, min_periods=3, center=True).median()
    df['deltaT_roll'] = df['deltaT'].rolling(window=roll_window, min_periods=3, center=True).median()
    return df

def detect_cycles(df: pd.DataFrame, soc_col:str='battery_SOC', smooth_window:int=10, min_dod:float=3.0) -> pd.DataFrame:
    if soc_col not in df.columns:
        raise ValueError(f"CSV must contain '{soc_col}' column.")
    soc = df[soc_col].rolling(window=smooth_window, min_periods=3, center=True).mean()
    ds = soc.diff()
    sign = np.sign(ds.fillna(0))
    # Turning points where sign changes (exclude flats)
    turn_idx = np.where(np.diff(np.sign(sign)) != 0)[0] + 1

    rows = []
    for i in range(len(turn_idx) - 1):
        a, b = turn_idx[i], turn_idx[i+1]
        start_soc, end_soc = soc.iloc[a], soc.iloc[b]
        if pd.isna(start_soc) or pd.isna(end_soc):
            continue
        dod = float(abs(end_soc - start_soc))
        if dod < min_dod:
            continue
        direction = 'charge' if end_soc > start_soc else 'discharge'
        rows.append({
            'start_time': df.loc[a, 'datetime'],
            'end_time': df.loc[b, 'datetime'],
            'start_SOC(%)': round(float(start_soc), 2),
            'end_SOC(%)': round(float(end_soc), 2),
            'DOD(%)': round(dod, 2),
            'direction': direction
        })
    return pd.DataFrame(rows)

def save_tables(outdir: Path, summary: pd.DataFrame, cycles: pd.DataFrame, dod_hist: pd.DataFrame):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "battery_feature_summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
    (outdir / "battery_cycles_table.csv").write_text(cycles.to_csv(index=False), encoding="utf-8")
    (outdir / "battery_dod_histogram.csv").write_text(dod_hist.to_csv(index=False), encoding="utf-8")

def plot_dod_hist(cycles: pd.DataFrame, outdir: Path):
    bins = np.arange(0, 105, 5)
    plt.figure(figsize=(10,5))
    plt.hist(cycles['DOD(%)'] if not cycles.empty else [], bins=bins, edgecolor='black')
    plt.title("DOD Histogram (Cycle Amplitudes, %SOC)")
    plt.xlabel("DOD (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "plot_dod_hist.png", dpi=150)
    plt.close()

def plot_deltaV(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12,5))
    plt.plot(df['datetime'], df['deltaV_roll'])
    plt.title("ΔV Trend (max_cell_voltage - min_cell_voltage, rolling median)")
    plt.xlabel("Time")
    plt.ylabel("ΔV (V)")
    plt.tight_layout()
    plt.savefig(outdir / "plot_deltaV_trend.png", dpi=150)
    plt.close()

def plot_deltaT(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12,5))
    plt.plot(df['datetime'], df['deltaT_roll'])
    plt.title("ΔT Trend (max_cell_temperature - min_cell_temperature, rolling median)")
    plt.xlabel("Time")
    plt.ylabel("ΔT (°C)")
    plt.tight_layout()
    plt.savefig(outdir / "plot_deltaT_trend.png", dpi=150)
    plt.close()

def plot_cum_Ah(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12,5))
    plt.plot(df['datetime'], df['cum_Ah'])
    plt.title("Cumulative Ah Throughput (|I| integrated)")
    plt.xlabel("Time")
    plt.ylabel("Ah")
    plt.tight_layout()
    plt.savefig(outdir / "plot_cum_Ah.png", dpi=150)
    plt.close()

def plot_cum_Wh(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12,5))
    plt.plot(df['datetime'], df['cum_Wh']/1000.0)
    plt.title("Cumulative Energy Throughput (|P| integrated)")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.tight_layout()
    plt.savefig(outdir / "plot_cum_Wh.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Battery Feature Extractor")
    parser.add_argument("--csv", type=Path, required=True, help="Path to input CSV (e.g., wf512.csv)")
    parser.add_argument("--outdir", type=Path, default=Path("./battery_features_out"), help="Output directory")
    parser.add_argument("--smooth", type=int, default=10, help="SOC smoothing window (samples)")
    parser.add_argument("--min-dod", type=float, default=3.0, help="Min DOD (%%SOC) to count as a cycle")
    parser.add_argument("--roll", type=int, default=30, help="Rolling window (samples) for ΔV/ΔT trends")
    args = parser.parse_args()

    df = load_data(args.csv)
    df = add_time_deltas(df)
    df = add_power_and_throughput(df)
    df = add_deltaV_deltaT(df, roll_window=args.roll)
    cycles_df = detect_cycles(df, smooth_window=args.smooth, min_dod=args.min_dod)

    # Summary metrics
    total_hours = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 3600.0
    summary = pd.DataFrame([{
        "time_range_start": df['datetime'].iloc[0],
        "time_range_end": df['datetime'].iloc[-1],
        "duration_hours": round(total_hours, 2),
        "median_interval_seconds": float(df['delta_t_s'].median(skipna=True)) if pd.notna(df['delta_t_s'].median(skipna=True)) else None,
        "total_throughput_Ah(abs)": round(df['abs_Ah'].sum(), 2),
        "total_charge_Ah": round(df['chg_Ah'].sum(), 2),
        "total_discharge_Ah": round(df['dch_Ah'].sum(), 2),
        "total_throughput_kWh(abs)": round(df['abs_Wh'].sum()/1000.0, 2),
        "total_charge_kWh": round(df['chg_Wh'].sum()/1000.0, 2),
        "total_discharge_kWh": round(df['dch_Wh'].sum()/1000.0, 2),
        "cycle_count(>=min_dod%)": int(len(cycles_df)),
        "avg_DOD_per_cycle_%": round(float(cycles_df['DOD(%)'].mean()), 2) if not cycles_df.empty else None,
        "median_DOD_per_cycle_%": round(float(cycles_df['DOD(%)'].median()), 2) if not cycles_df.empty else None
    }])

    # DOD histogram table
    bins = np.arange(0, 105, 5)
    hist_counts, bin_edges = np.histogram(cycles_df['DOD(%)'] if not cycles_df.empty else [], bins=bins)
    dod_hist_df = pd.DataFrame({
        "DOD_bin_%": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist_counts))],
        "count": hist_counts
    })

    # Save tables
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "battery_feature_summary.csv", index=False)
    cycles_df.to_csv(outdir / "battery_cycles_table.csv", index=False)
    dod_hist_df.to_csv(outdir / "battery_dod_histogram.csv", index=False)

    # Plots
    plot_dod_hist(cycles_df, outdir)
    plot_deltaV(df, outdir)
    plot_deltaT(df, outdir)
    plot_cum_Ah(df, outdir)
    plot_cum_Wh(df, outdir)

    print("Saved CSVs and plots to:", str(outdir.resolve()))

if __name__ == "__main__":
    main()
