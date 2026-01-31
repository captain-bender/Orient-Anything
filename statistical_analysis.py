import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Load the comparison data
df = pd.read_csv('benchmark_results/comparison.csv')

# Convert front_identified column to boolean (handle string values)
df['front_identified'] = df['front_identified'].astype(str).str.lower() == 'true'

# Extract axis_error column
axis_errors = df['axis_error'].dropna()

print("=" * 60)
print("STATISTICAL ANALYSIS OF AXIS ERROR")
print("=" * 60)

# Dictionary to store all statistics for JSON export
statistics_dict = {
    "metadata": {
        "analysis_date": datetime.now().isoformat(),
        "total_samples": len(axis_errors),
        "data_source": "benchmark_results/comparison.csv",
        "column_analyzed": "axis_error"
    }
}

# Basic Statistics
print("\n1. DESCRIPTIVE STATISTICS")
print("-" * 60)
mean_val = axis_errors.mean()
median_val = axis_errors.median()
std_val = axis_errors.std()
var_val = axis_errors.var()
min_val = axis_errors.min()
max_val = axis_errors.max()
range_val = max_val - min_val

print(f"Count:              {len(axis_errors)}")
print(f"Mean:               {mean_val:.4f}°")
print(f"Median:             {median_val:.4f}°")
print(f"Std Deviation:      {std_val:.4f}°")
print(f"Variance:           {var_val:.4f}")
print(f"Min:                {min_val:.4f}°")
print(f"Max:                {max_val:.4f}°")
print(f"Range:              {range_val:.4f}°")

statistics_dict["descriptive_statistics"] = {
    "count": int(len(axis_errors)),
    "mean": round(float(mean_val), 4),
    "median": round(float(median_val), 4),
    "std_deviation": round(float(std_val), 4),
    "variance": round(float(var_val), 4),
    "min": round(float(min_val), 4),
    "max": round(float(max_val), 4),
    "range": round(float(range_val), 4)
}

# Quartiles
print("\n2. QUARTILE ANALYSIS")
print("-" * 60)
q1_val = axis_errors.quantile(0.25)
q2_val = axis_errors.quantile(0.50)
q3_val = axis_errors.quantile(0.75)
iqr_val = q3_val - q1_val

print(f"Q1 (25th percentile): {q1_val:.4f}°")
print(f"Q2 (50th percentile): {q2_val:.4f}°")
print(f"Q3 (75th percentile): {q3_val:.4f}°")
print(f"IQR (Q3 - Q1):        {iqr_val:.4f}°")

statistics_dict["quartile_analysis"] = {
    "q1": round(float(q1_val), 4),
    "q2": round(float(q2_val), 4),
    "q3": round(float(q3_val), 4),
    "iqr": round(float(iqr_val), 4)
}

# Skewness and Kurtosis
print("\n3. DISTRIBUTION SHAPE")
print("-" * 60)
skewness_val = stats.skew(axis_errors)
kurtosis_val = stats.kurtosis(axis_errors)

print(f"Skewness:           {skewness_val:.4f}")
print(f"Kurtosis:           {kurtosis_val:.4f}")

statistics_dict["distribution_shape"] = {
    "skewness": round(float(skewness_val), 4),
    "kurtosis": round(float(kurtosis_val), 4)
}

# Normality Test (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(axis_errors)
print("\n4. NORMALITY TEST (Shapiro-Wilk)")
print("-" * 60)
print(f"Test Statistic:     {shapiro_stat:.4f}")
print(f"P-value:            {shapiro_p:.4f}")
print(f"Result:             {'Data appears normal' if shapiro_p > 0.05 else 'Data does not appear normal'} (α=0.05)")

statistics_dict["normality_test"] = {
    "test": "Shapiro-Wilk",
    "statistic": round(float(shapiro_stat), 4),
    "p_value": round(float(shapiro_p), 4),
    "is_normal": bool(shapiro_p > 0.05),
    "alpha": 0.05
}

# Confidence Intervals
print("\n5. CONFIDENCE INTERVALS (95%)")
print("-" * 60)
sem = stats.sem(axis_errors)
ci = stats.t.interval(0.95, len(axis_errors)-1, loc=mean_val, scale=sem)
print(f"95% CI for Mean:    [{ci[0]:.4f}°, {ci[1]:.4f}°]")

statistics_dict["confidence_intervals"] = {
    "confidence_level": 0.95,
    "mean_ci_lower": round(float(ci[0]), 4),
    "mean_ci_upper": round(float(ci[1]), 4)
}

# Axis Error Distribution Bins
print("\n6. AXIS ERROR DISTRIBUTION")
print("-" * 60)
bins = [0, 1, 2, 3, 4, 5, float('inf')]
labels = ['0-1deg', '1-2deg', '2-3deg', '3-4deg', '4-5deg', '>5deg']
binned = pd.cut(axis_errors, bins=bins, labels=labels)
distribution = binned.value_counts().sort_index()
print("Axis Error Range | Count | Percentage")
distribution_dict = {}
for label, count in distribution.items():
    percentage = (count / len(axis_errors)) * 100
    print(f"{label:>11} | {count:5d} | {percentage:6.2f}%")
    distribution_dict[str(label)] = {
        "count": int(count),
        "percentage": round(float(percentage), 2)
    }

statistics_dict["error_distribution"] = distribution_dict

# Correlation with confidence_score
print("\n7. CORRELATION WITH CONFIDENCE SCORE")
print("-" * 60)
correlation = df['axis_error'].corr(df['confidence_score'])
print(f"Pearson Correlation: {correlation:.4f}")

statistics_dict["correlation_analysis"] = {
    "variable": "confidence_score",
    "correlation_type": "Pearson",
    "correlation_coefficient": round(float(correlation), 4)
}

# Save statistics to JSON
print("\n8. SAVING STATISTICS TO JSON...")
print("-" * 60)
with open('benchmark_results/axis_error_statistics.json', 'w') as f:
    json.dump(statistics_dict, f, indent=2)
print("Statistics saved to: benchmark_results/axis_error_statistics.json")

# Create individual visualizations
print("\n9. CREATING INDIVIDUAL VISUALIZATIONS...")
print("-" * 60)

# 1. Histogram with normal distribution
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(axis_errors, bins=15, density=True, alpha=0.7, color='blue', edgecolor='black')
mu, sigma = mean_val, std_val
x = np.linspace(min_val, max_val, 100)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
ax1.set_xlabel('Axis Error (deg)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Histogram with Normal Distribution Fit', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_results/01_histogram.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/01_histogram.png")
plt.close()

# 2. Box plot
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.boxplot(axis_errors, vert=True)
ax2.set_ylabel('Axis Error (deg)', fontsize=12)
ax2.set_title('Box Plot', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_results/02_boxplot.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/02_boxplot.png")
plt.close()

# 3. Q-Q plot for normality assessment
fig3, ax3 = plt.subplots(figsize=(8, 6))
stats.probplot(axis_errors, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normality Assessment)', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_results/03_qq_plot.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/03_qq_plot.png")
plt.close()

# 4. Scatter plot: Axis Error vs Confidence Score
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.scatter(df['confidence_score'], df['axis_error'], alpha=0.6, s=50)
ax4.set_xlabel('Confidence Score', fontsize=12)
ax4.set_ylabel('Axis Error (deg)', fontsize=12)
ax4.set_title(f'Axis Error vs Confidence Score\n(Correlation: {correlation:.4f})', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)

# Add trend line
z = np.polyfit(df['confidence_score'], df['axis_error'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df['confidence_score'].min(), df['confidence_score'].max(), 100)
ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
ax4.legend()
plt.tight_layout()
plt.savefig('benchmark_results/04_correlation_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/04_correlation_scatter.png")
plt.close()

# 5. Axis Error distribution bar chart
fig5, ax5 = plt.subplots(figsize=(10, 6))
distribution.plot(kind='bar', ax=ax5, color='steelblue', edgecolor='black')
ax5.set_xlabel('Axis Error Range (deg)', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.set_title('Axis Error Distribution by Range', fontsize=14, fontweight='bold')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
ax5.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('benchmark_results/05_error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/05_error_distribution.png")
plt.close()

# 6. Front Identified Pie Chart
fig6, ax6 = plt.subplots(figsize=(8, 8))
front_identified_counts = df['front_identified'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for True, Red for False

# Get counts safely
true_count = front_identified_counts.get(True, 0)
false_count = front_identified_counts.get(False, 0)

labels = [f'Front Identified\n{true_count} predictions',
          f'Front Not Identified\n{false_count} predictions']
sizes = [true_count, false_count]

# Only create pie chart if we have at least one non-zero value
if sum(sizes) > 0:
    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=90, 
                                         textprops={'fontsize': 11})
    
    # Make percentage text bold and visible
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

ax6.set_title('Front Identification Status Distribution', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('benchmark_results/06_front_identified_pie.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/06_front_identified_pie.png")
plt.close()

# Add front identified statistics to the JSON
front_identified_stats = {
    "correctly_identified": int(front_identified_counts.get(True, 0)),
    "not_identified": int(front_identified_counts.get(False, 0)),
    "correct_percentage": round(float(front_identified_counts.get(True, 0) / len(df) * 100), 2),
    "incorrect_percentage": round(float(front_identified_counts.get(False, 0) / len(df) * 100), 2)
}
statistics_dict["front_identification_status"] = front_identified_stats

# 7. Quality Distribution Pie Chart
fig7, ax7 = plt.subplots(figsize=(8, 8))

# Categorize based on quality ranges
excellent_count = len(df[df['axis_error'] <= 3])  # 0-3 degrees
very_good_count = len(df[(df['axis_error'] > 3) & (df['axis_error'] <= 5)])  # 3-5 degrees
good_count = len(df[df['axis_error'] > 5])  # Above 5 degrees

sizes = [excellent_count, very_good_count, good_count]
labels = [f'Excellent 0-3deg\n{excellent_count} predictions',
          f'Very Good 3-5deg\n{very_good_count} predictions',
          f'Good >5deg\n{good_count} predictions']
colors = ['#27ae60', '#f39c12', '#e74c3c']  # Green, Orange, Red

if sum(sizes) > 0:
    wedges, texts, autotexts = ax7.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         textprops={'fontsize': 11})
    
    # Make percentage text bold and visible
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

ax7.set_title('Quality Distribution by Axis Error Range', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('benchmark_results/07_quality_distribution_pie.png', dpi=300, bbox_inches='tight')
print("Saved: benchmark_results/07_quality_distribution_pie.png")
plt.close()

# Add quality distribution statistics to the JSON
quality_stats = {
    "excellent_0_to_3deg": {
        "count": int(excellent_count),
        "percentage": round(float(excellent_count / len(df) * 100), 2)
    },
    "very_good_3_to_5deg": {
        "count": int(very_good_count),
        "percentage": round(float(very_good_count / len(df) * 100), 2)
    },
    "good_above_5deg": {
        "count": int(good_count),
        "percentage": round(float(good_count / len(df) * 100), 2)
    }
}
statistics_dict["quality_distribution"] = quality_stats

# Update the JSON file with the new statistics
with open('benchmark_results/axis_error_statistics.json', 'w') as f:
    json.dump(statistics_dict, f, indent=2)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nGenerated files:")
print("  - benchmark_results/axis_error_statistics.json")
print("  - benchmark_results/01_histogram.png")
print("  - benchmark_results/02_boxplot.png")
print("  - benchmark_results/03_qq_plot.png")
print("  - benchmark_results/04_correlation_scatter.png")
print("  - benchmark_results/05_error_distribution.png")
print("  - benchmark_results/06_front_identified_pie.png")
print("  - benchmark_results/07_quality_distribution_pie.png")
