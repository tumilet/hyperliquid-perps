#!/usr/bin/env python3
"""
Plot Open Interest, Perp Equity, and their ratio with dual y-axis.
Simplified version - only chart generation logic.
"""

import json
import csv
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def fetch_json_data(url):
    """Fetch JSON data from URL."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode('utf-8'))

def parse_data(json_data, start_date=None):
    """Parse the historical data from JSON, taking one data point per day."""
    data_obj = json_data.get('data', {})
    columns = data_obj.get('columns', [])
    rows = data_obj.get('data', [])

    # Find column indices
    timestamp_idx = columns.index('timestamp')
    oi_idx = columns.index('totalOpenInterest')
    equity_idx = columns.index('perpEquity')

    # Dictionary to store last data point for each date
    daily_data = {}

    for row in rows:
        ts = datetime.fromisoformat(row[timestamp_idx].replace('Z', '+00:00'))

        # Filter by start date if provided
        if start_date and ts.replace(tzinfo=None) < start_date:
            continue

        oi = float(row[oi_idx])
        equity = float(row[equity_idx])
        ratio = oi / equity if equity > 0 else 0

        # Use date as key, keep last entry for each day
        date_key = ts.date()
        daily_data[date_key] = {
            'timestamp': ts,
            'oi': oi,
            'equity': equity,
            'ratio': ratio
        }

    # Sort by date and extract arrays
    sorted_dates = sorted(daily_data.keys())
    timestamps = [daily_data[d]['timestamp'] for d in sorted_dates]
    open_interest = [daily_data[d]['oi'] for d in sorted_dates]
    perp_equity = [daily_data[d]['equity'] for d in sorted_dates]
    ratios = [daily_data[d]['ratio'] for d in sorted_dates]

    return timestamps, open_interest, perp_equity, ratios

def plot_dual_axis_chart(timestamps, open_interest, perp_equity, ratios):
    """Create dual y-axis chart."""
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Colors
    color1 = '#2E86DE'  # Blue - Open Interest
    color2 = '#10AC84'  # Green - Perp Equity
    color3 = '#000000'  # Black - Ratio

    # Left y-axis setup
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('USD (Billions)', fontsize=12, fontweight='bold', color='black')

    # Plot OI and Equity on left axis
    line1 = ax1.plot(timestamps, [oi/1e9 for oi in open_interest],
                     color=color1, linewidth=2, label='Total Open Interest',
                     marker='o', markersize=2, markerfacecolor=color1, markeredgewidth=0)

    line2 = ax1.plot(timestamps, [eq/1e9 for eq in perp_equity],
                     color=color2, linewidth=2, label='Perp Equity',
                     marker='o', markersize=2, markerfacecolor=color2, markeredgewidth=0)

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}B'))

    # Right y-axis setup for ratio
    ax2 = ax1.twinx()
    ax2.set_ylabel('Leverage Ratio (OI / Equity)', fontsize=12, fontweight='bold', color=color3)

    line3 = ax2.plot(timestamps, ratios,
                     color=color3, linewidth=2, label='OI/Equity Ratio',
                     marker='o', markersize=2, markerfacecolor=color3, markeredgewidth=0,
                     linestyle='--')

    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}x'))

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.set_xlim(left=timestamps[0])  # Start x-axis from first data point
    plt.xticks(rotation=45, ha='right')

    # Title and grid
    ax1.set_title('Hyperliquid: Open Interest, Perp Equity & Leverage Ratio',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)

    # Current value labels at end of lines
    last_date = timestamps[-1]
    last_oi = open_interest[-1] / 1e9
    last_equity = perp_equity[-1] / 1e9
    last_ratio = ratios[-1]

    ax1.annotate(f'${last_oi:.2f}B',
                xy=(last_date, last_oi),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color=color1,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color1, alpha=0.8))

    ax1.annotate(f'${last_equity:.2f}B',
                xy=(last_date, last_equity),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color=color2,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color2, alpha=0.8))

    ax2.annotate(f'{last_ratio:.2f}x',
                xy=(last_date, last_ratio),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color=color3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color3, alpha=0.8))

    # Labels for Oct 10 and Oct 11, 2025
    target_dates = [
        datetime(2025, 10, 10),
        datetime(2025, 10, 11)
    ]

    for target_date in target_dates:
        # Find closest data point
        closest_idx = min(range(len(timestamps)),
                         key=lambda i: abs((timestamps[i].replace(tzinfo=None) - target_date).total_seconds()))

        date_oi = open_interest[closest_idx] / 1e9
        date_equity = perp_equity[closest_idx] / 1e9
        date_ratio = ratios[closest_idx]
        date_ts = timestamps[closest_idx]

        # Vertical line marker
        ax1.axvline(x=date_ts, color='gray', linestyle=':', alpha=0.3, linewidth=1)

        # Labels for each metric
        ax1.annotate(f'${date_oi:.2f}B',
                    xy=(date_ts, date_oi),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color1,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color1, alpha=0.8),
                    ha='center')

        ax1.annotate(f'${date_equity:.2f}B',
                    xy=(date_ts, date_equity),
                    xytext=(0, -15), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color2,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color2, alpha=0.8),
                    ha='center')

        ax2.annotate(f'{date_ratio:.2f}x',
                    xy=(date_ts, date_ratio),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color3,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color3, alpha=0.8),
                    ha='center')

    # Statistics box
    stats_text = (
        f'Current Stats:\n'
        f'OI: ${open_interest[-1]/1e9:.2f}B\n'
        f'Equity: ${perp_equity[-1]/1e9:.2f}B\n'
        f'Ratio: {ratios[-1]:.2f}x\n'
        f'\n'
        f'Ratio Range:\n'
        f'Min: {min(ratios):.2f}x\n'
        f'Max: {max(ratios):.2f}x\n'
        f'Avg: {sum(ratios)/len(ratios):.2f}x'
    )

    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    output_file = 'oi_equity_ratio_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved as {output_file}")
    plt.close()

def save_to_csv(timestamps, open_interest, perp_equity, ratios):
    """Save data to CSV file."""
    filename = 'oi_equity_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'totalOpenInterest', 'perpEquity', 'ratio'])
        for i in range(len(timestamps)):
            writer.writerow([
                timestamps[i].isoformat(),
                open_interest[i],
                perp_equity[i],
                ratios[i]
            ])
    print(f"✓ Data saved to {filename}")

def main():
    url = "https://dw3ji7n7thadj.cloudfront.net/aggregator/stats/metrics_perp_positions_charts_data.json"

    print("Fetching data...")
    json_data = fetch_json_data(url)

    # Filter data from June 1, 2025 onwards
    start_date = datetime(2025, 6, 1)
    timestamps, open_interest, perp_equity, ratios = parse_data(json_data, start_date=start_date)

    print(f"Loaded {len(timestamps)} data points")
    print(f"Period: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}")

    print("Saving to CSV...")
    save_to_csv(timestamps, open_interest, perp_equity, ratios)

    print("Generating chart...")
    plot_dual_axis_chart(timestamps, open_interest, perp_equity, ratios)

if __name__ == "__main__":
    main()
