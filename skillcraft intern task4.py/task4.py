import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import os
import sys

# ============================================================
# TASK 04 - TRAFFIC ACCIDENT DATA ANALYSIS
# ============================================================

# Usage:
# python3 traffic_accident_analysis.py accident_dataset.csv

# ------------------------------------------------------------
# 1. Read the dataset
# ------------------------------------------------------------

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "accident_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of records:", len(df))
print("\nFirst 5 records:")
print(df.head())

# ------------------------------------------------------------
# 2. Basic information
# ------------------------------------------------------------

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# 3. Create output folder
# ------------------------------------------------------------

output_folder = "output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ------------------------------------------------------------
# 4. Convert Date and Time
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%H:%M",
    errors="coerce"
)

df["Hour"] = df["Time"].dt.hour

# ------------------------------------------------------------
# 5. Create Time of Day category
# ------------------------------------------------------------

def get_time_period(hour):

    if 5 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 21:
        return "Evening"

    else:
        return "Night"


df["Time_of_Day"] = df["Hour"].apply(get_time_period)

# ------------------------------------------------------------
# 6. Accidents by Weather
# ------------------------------------------------------------

weather_counts = df["Weather"].value_counts()

plt.figure(figsize=(10, 6))

weather_counts.plot(kind="bar")

plt.title("Traffic Accidents by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/01_accidents_by_weather.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 7. Accidents by Road Condition
# ------------------------------------------------------------

road_counts = df["Road_Condition"].value_counts()

plt.figure(figsize=(10, 6))

road_counts.plot(kind="bar")

plt.title("Traffic Accidents by Road Condition")
plt.xlabel("Road Condition")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/02_accidents_by_road_condition.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 8. Accidents by Time of Day
# ------------------------------------------------------------

time_counts = df["Time_of_Day"].value_counts()

order = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
]

time_counts = time_counts.reindex(order)

plt.figure(figsize=(10, 6))

time_counts.plot(kind="bar")

plt.title("Traffic Accidents by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/03_accidents_by_time_of_day.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 9. Accident Severity
# ------------------------------------------------------------

severity_counts = df["Severity"].value_counts()

plt.figure(figsize=(10, 6))

severity_counts.plot(kind="bar")

plt.title("Traffic Accidents by Severity")
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/04_accidents_by_severity.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 10. Accident Hotspots
# ------------------------------------------------------------

hotspots = df["Location"].value_counts().head(10)

plt.figure(figsize=(10, 6))

hotspots.sort_values().plot(kind="barh")

plt.title("Top 10 Traffic Accident Hotspots")
plt.xlabel("Number of Accidents")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/05_top_accident_hotspots.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 11. Contributing Factors
# ------------------------------------------------------------

cause_counts = df["Cause"].value_counts()

plt.figure(figsize=(10, 6))

cause_counts.sort_values().plot(kind="barh")

plt.title("Major Contributing Factors to Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Contributing Factor")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/06_contributing_factors.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 12. Weather vs Severity Heatmap
# ------------------------------------------------------------

weather_severity = pd.crosstab(
    df["Weather"],
    df["Severity"]
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    weather_severity,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Weather Condition vs Accident Severity")
plt.xlabel("Severity")
plt.ylabel("Weather Condition")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/07_weather_vs_severity.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 13. Road Condition vs Severity
# ------------------------------------------------------------

road_severity = pd.crosstab(
    df["Road_Condition"],
    df["Severity"]
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    road_severity,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("Road Condition vs Accident Severity")
plt.xlabel("Severity")
plt.ylabel("Road Condition")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/08_road_condition_vs_severity.png",
    dpi=200
)

plt.show()

# ------------------------------------------------------------
# 14. Accident Hotspot Map
# ------------------------------------------------------------

map_data = df.dropna(
    subset=["Latitude", "Longitude"]
)

if len(map_data) > 0:

    center_lat = map_data["Latitude"].mean()
    center_lon = map_data["Longitude"].mean()

    accident_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10
    )

    heat_data = map_data[
        ["Latitude", "Longitude"]
    ].values.tolist()

    HeatMap(
        heat_data,
        radius=15,
        blur=10
    ).add_to(accident_map)

    accident_map.save(
        f"{output_folder}/09_accident_hotspot_map.html"
    )

# ------------------------------------------------------------
# 15. Display important findings
# ------------------------------------------------------------

print("\n============================================")
print("TRAFFIC ACCIDENT ANALYSIS RESULTS")
print("============================================")

print(
    "\nTotal accidents:",
    len(df)
)

print(
    "\nMost common weather condition:",
    df["Weather"].value_counts().idxmax()
)

print(
    "Most common road condition:",
    df["Road_Condition"].value_counts().idxmax()
)

print(
    "Most dangerous time period:",
    df["Time_of_Day"].value_counts().idxmax()
)

print(
    "Top accident hotspot:",
    df["Location"].value_counts().idxmax()
)

print(
    "Most common contributing factor:",
    df["Cause"].value_counts().idxmax()
)

print(
    "Most common accident severity:",
    df["Severity"].value_counts().idxmax()
)

print("\nAnalysis completed successfully!")

print(
    f"\nAll graphs and the hotspot map are saved inside the "
    f"'{output_folder}' folder."
)