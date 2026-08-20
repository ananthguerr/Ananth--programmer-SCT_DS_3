# Task 04 - Traffic Accident Analysis

## Objective
Analyze traffic accident data to identify patterns related to:
- Road conditions
- Weather
- Time of day
- Accident hotspots
- Contributing factors
- Accident severity

## Files
- `accident_dataset.csv` - sample dataset ready for the project
- `traffic_accident_analysis.py` - source code
- `requirements.txt` - required Python libraries
- `README.md` - project instructions

## Dataset Format
The Python program is designed to accept the supplied CSV dataset format.

Required columns:

```text
Accident_ID
Date
Time
Weather
Road_Condition
Location
Latitude
Longitude
Severity
Vehicles_Involved
Cause
```

You can replace `accident_dataset.csv` with another CSV dataset only if it contains these required columns.

## Run on Mac / VS Code

Open Terminal in the project folder:

```bash
python3 -m pip install -r requirements.txt
python3 traffic_accident_analysis.py accident_dataset.csv
```

The program also accepts a different CSV file:

```bash
python3 traffic_accident_analysis.py your_dataset.csv
```

To choose a custom output folder:

```bash
python3 traffic_accident_analysis.py accident_dataset.csv --output results
```

## Output
The program creates an `output` folder containing:
- Accidents by weather chart
- Accidents by road condition chart
- Accidents by time of day chart
- Accident severity chart
- Top accident hotspots chart
- Contributing factors chart
- Weather vs severity heatmap
- Road condition vs severity heatmap
- Interactive accident hotspot map
- Text analysis summary

## Internship Presentation
Recommended project title:

**Traffic Accident Pattern and Hotspot Analysis Using Python**

Tools:
Python, Pandas, NumPy, Matplotlib, Seaborn and Folium.

This project performs exploratory data analysis and visualization. It does not claim that a statistical association automatically proves causation.
