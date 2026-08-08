import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# SAMPLE DATASET
# -------------------------------

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

sales = [
    120, 145, 135, 180, 210, 195,
    230, 250, 220, 275, 300, 340
]

# -------------------------------
# CREATE BAR CHART
# -------------------------------

plt.figure(figsize=(12, 7))

bars = plt.bar(
    months,
    sales,
    width=0.65,
    edgecolor="black",
    linewidth=1
)

# -------------------------------
# ADD VALUES ON TOP OF BARS
# -------------------------------

for bar, value in zip(bars, sales):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"₹{value}L",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# -------------------------------
# TITLE AND LABELS
# -------------------------------

plt.title(
    "Monthly Sales Performance – 2025",
    fontsize=20,
    fontweight="bold",
    pad=20
)

plt.xlabel(
    "Month",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Sales (₹ Lakhs)",
    fontsize=13,
    fontweight="bold"
)

# -------------------------------
# GRID
# -------------------------------

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

# Start Y-axis from zero
plt.ylim(0, 380)

# -------------------------------
# CLEAN APPEARANCE
# -------------------------------

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()

# Display chart
plt.show()
