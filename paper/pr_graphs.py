import matplotlib.pyplot as plt
import csv

# Sample code to read data from a CSV file without using pandas
def read_csv_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        models = []
        p_scores_AIDA_A = []
        p_scores_AIDA_B = []
        p_scores_AIDA_C = []
        r_scores_AIDA_A = []
        r_scores_AIDA_B = []
        r_scores_AIDA_C = []

        for row in reader:
            models.append(row['model'])
            p_scores_AIDA_A.append(float(row['AIDA-A precision']))
            p_scores_AIDA_B.append(float(row['AIDA-B precision']))
            p_scores_AIDA_C.append(float(row['AIDA-C precision']))
            r_scores_AIDA_A.append(float(row['AIDA-A recall']))
            r_scores_AIDA_B.append(float(row['AIDA-B recall']))
            r_scores_AIDA_C.append(float(row['AIDA-C recall']))

        return models, p_scores_AIDA_A, p_scores_AIDA_B, p_scores_AIDA_C, r_scores_AIDA_A, r_scores_AIDA_B, r_scores_AIDA_C

file_path = 'table1-micropr.txt'  # Replace with your CSV file path
models, p_scores_AIDA_A, p_scores_AIDA_B, p_scores_AIDA_C, r_scores_AIDA_A, r_scores_AIDA_B, r_scores_AIDA_C = read_csv_data(file_path)

#max_f1_score = max(max(f1_scores_AIDA_A), max(f1_scores_AIDA_B), max(f1_scores_AIDA_C))
max_score = 100

bar_width = 0.35  # Width of the bars

# Creating the bar graphs
fig, axes = plt.subplots(1, 3, figsize=(8, 5.5), sharey=True)

# Helper function to plot bars
def plot_bars(ax, recall_scores, precision_scores, title, show_legend=False):
    # Calculate positions for each group of bars
    indices = range(len(recall_scores))
    ax.bar([i - bar_width/2 for i in indices], precision_scores, bar_width, label='Micro Precision', color='blue')
    ax.bar([i + bar_width/2 for i in indices], recall_scores, bar_width, label='Micro Recall', color='red')

    ax.set_title(title)
    ax.set_xticks(indices)
    ax.set_xticklabels(models, rotation=90)
    ax.set_ylim([0, max_score])
    ax.grid(True, linestyle='--', color='gray')
    if show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=1, framealpha=0.25, fontsize='small')

# AIDA-A
plot_bars(axes[0], r_scores_AIDA_A, p_scores_AIDA_A, 'AIDA-A')

# AIDA-B
plot_bars(axes[1], r_scores_AIDA_B, p_scores_AIDA_B, 'AIDA-B')

# AIDA-C
plot_bars(axes[2], r_scores_AIDA_C, p_scores_AIDA_C, 'AIDA-C', show_legend=True)


# Overall title and layout adjustments
#plt.suptitle('Recall and Precision Comparison Across Models for AIDA-A, AIDA-B, and AIDA-C')
plt.tight_layout()

# Display the graphs
plt.show()
