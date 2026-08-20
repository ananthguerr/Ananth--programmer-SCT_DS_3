"""
TASK 04 - Traffic Accident Data Analysis

Usage:
    python traffic_accident_analysis.py accident_dataset.csv

The script accepts a CSV dataset containing the columns:
Accident_ID, Date, Time, Weather, Road_Condition, Location,
Latitude, Longitude, Severity, Vehicles_Involved, Cause

It produces:
    output/01_accidents_by_weather.png
    output/02_accidents_by_road_condition.png
    output/03_accidents_by_time_of_day.png
    output/04_accidents_by_severity.png
    output/05_top_accident_hotspots.png
    output/06_contributing_factors.png
    output/07_weather_vs_severity.png
    output/08_road_condition_vs_severity.png
    output/09_hotspot_map.html
    output/analysis_summary.txt
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REQUIRED_COLUMNS = [
    "Accident_ID", "Date", "Time", "Weather", "Road_Condition",
    "Location", "Latitude", "Longitude", "Severity",
    "Vehicles_Involved", "Cause"
]

def load_and_validate_data(file_path):
    """Load the given CSV and validate the required columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Time"] = df["Time"].astype(str)

    # Convert time into hour safely
    df["Hour"] = pd.to_datetime(
        df["Time"], format="%H:%M", errors="coerce"
    ).dt.hour

    # If time contains seconds, try a second parsing method
    bad_hours = df["Hour"].isna()
    if bad_hours.any():
        df.loc[bad_hours, "Hour"] = pd.to_datetime(
            df.loc[bad_hours, "Time"], errors="coerce"
        ).dt.hour

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Vehicles_Involved"] = pd.to_numeric(
        df["Vehicles_Involved"], errors="coerce"
    )

    df["Time_of_Day"] = pd.cut(
        df["Hour"],
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"]
    )

    return df

def save_bar_chart(series, title, xlabel, filename, horizontal=False):
    plt.figure(figsize=(10, 6))
    if horizontal:
        series.sort_values().plot(kind="barh")
    else:
        series.plot(kind="bar")
        plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def create_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Weather
    weather_counts = df["Weather"].value_counts()
    save_bar_chart(
        weather_counts,
        "Accidents by Weather Condition",
        "Weather Condition",
        os.path.join(output_dir, "01_accidents_by_weather.png")
    )

    # 2. Road condition
    road_counts = df["Road_Condition"].value_counts()
    save_bar_chart(
        road_counts,
        "Accidents by Road Condition",
        "Road Condition",
        os.path.join(output_dir, "02_accidents_by_road_condition.png")
    )

    # 3. Time of day
    time_counts = df["Time_of_Day"].value_counts().reindex(
        ["Night", "Morning", "Afternoon", "Evening", "Late Night"]
    ).dropna()
    save_bar_chart(
        time_counts,
        "Accidents by Time of Day",
        "Time of Day",
        os.path.join(output_dir, "03_accidents_by_time_of_day.png")
    )

    # 4. Severity
    severity_counts = df["Severity"].value_counts()
    save_bar_chart(
        severity_counts,
        "Accidents by Severity",
        "Severity",
        os.path.join(output_dir, "04_accidents_by_severity.png")
    )

    # 5. Hotspots
    hotspot_counts = df["Location"].value_counts().head(10)
    save_bar_chart(
        hotspot_counts,
        "Top 10 Accident Hotspots",
        "Accident Count",
        os.path.join(output_dir, "05_top_accident_hotspots.png"),
        horizontal=True
    )

    # 6. Contributing factors
    cause_counts = df["Cause"].value_counts().head(10)
    save_bar_chart(
        cause_counts,
        "Top Contributing Factors",
        "Contributing Factor",
        os.path.join(output_dir, "06_contributing_factors.png"),
        horizontal=True
    )

    # 7. Weather vs severity
    weather_severity = pd.crosstab(df["Weather"], df["Severity"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(weather_severity, annot=True, fmt="d", cmap="Blues")
    plt.title("Weather Condition vs Accident Severity")
    plt.xlabel("Severity")
    plt.ylabel("Weather")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "07_weather_vs_severity.png"),
        dpi=200
    )
    plt.close()

    # 8. Road condition vs severity
    road_severity = pd.crosstab(df["Road_Condition"], df["Severity"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(road_severity, annot=True, fmt="d", cmap="Oranges")
    plt.title("Road Condition vs Accident Severity")
    plt.xlabel("Severity")
    plt.ylabel("Road Condition")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "08_road_condition_vs_severity.png"),
        dpi=200
    )
    plt.close()

    # 9. Interactive hotspot map using Folium
    try:
        import folium
        from folium.plugins import HeatMap

        map_df = df.dropna(subset=["Latitude", "Longitude"]).copy()

        if not map_df.empty:
            center = [
                map_df["Latitude"].mean(),
                map_df["Longitude"].mean()
            ]

            accident_map = folium.Map(
                location=center,
                zoom_start=10,
                tiles="OpenStreetMap"
            )

            heat_data = map_df[["Latitude", "Longitude"]].values.tolist()
            HeatMap(heat_data, radius=15, blur=10).add_to(accident_map)

            accident_map.save(
                os.path.join(output_dir, "09_hotspot_map.html")
            )
    except ImportError:
        print("Folium is not installed. Skipping interactive map.")

def create_summary(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total = len(df)
    top_weather = df["Weather"].value_counts().idxmax()
    top_road = df["Road_Condition"].value_counts().idxmax()
    top_time = df["Time_of_Day"].value_counts().idxmax()
    top_location = df["Location"].value_counts().idxmax()
    top_cause = df["Cause"].value_counts().idxmax()
    top_severity = df["Severity"].value_counts().idxmax()

    summary = f"""
TASK 04 - TRAFFIC ACCIDENT ANALYSIS
===================================

Total accidents analysed : {total}

Most common weather      : {top_weather}
Most common road condition: {top_road}
Most common time period  : {top_time}
Top accident hotspot     : {top_location}
Top contributing factor  : {top_cause}
Most common severity     : {top_severity}

The generated charts can be used to identify:
1. Patterns related to weather.
2. Patterns related to road conditions.
3. High-risk times of day.
4. Accident hotspots.
5. Major contributing factors.
6. Relationships between conditions and accident severity.

Note:
This is an exploratory data-analysis project. Correlation in the dataset
does not automatically prove that a factor directly caused an accident.
"""

    with open(
        os.path.join(output_dir, "analysis_summary.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(summary.strip())

def main():
    parser = argparse.ArgumentParser(
        description="Analyse traffic accident data from a CSV file."
    )
    parser.add_argument(
        "dataset",
        help="Path to the accident CSV dataset"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where charts and reports will be saved"
    )

    args = parser.parse_args()

    try:
        df = load_and_validate_data(args.dataset)

        print(f"Loaded {len(df)} accident records.")
        print("\nColumns detected:")
        print(", ".join(df.columns))

        create_visualizations(df, args.output)
        create_summary(df, args.output)

        print(f"\nAnalysis completed successfully.")
        print(f"Results saved in: {args.output}/")

    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
