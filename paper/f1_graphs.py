import matplotlib.pyplot as plt
import csv

# Sample code to read data from a CSV file without using pandas
def read_csv_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        models = []
        f1_scores_AIDA_A = []
        f1_scores_AIDA_B = []
        f1_scores_AIDA_C = []

        for row in reader:
            models.append(row['model'])
            f1_scores_AIDA_A.append(float(row['AIDA-A']))
            f1_scores_AIDA_B.append(float(row['AIDA-B']))
            f1_scores_AIDA_C.append(float(row['AIDA-C']))

        return models, f1_scores_AIDA_A, f1_scores_AIDA_B, f1_scores_AIDA_C

file_path = 'normal-microf1.txt'  # Replace with your CSV file path
models, f1_scores_AIDA_A, f1_scores_AIDA_B, f1_scores_AIDA_C = read_csv_data(file_path)

max_f1_score = max(max(f1_scores_AIDA_A), max(f1_scores_AIDA_B), max(f1_scores_AIDA_C))
max_f1_score = 100

# Creating the bar graphs
fig, axes = plt.subplots(1, 3, figsize=(8, 3.75), sharey=True)

# AIDA-A
axes[0].bar(models, f1_scores_AIDA_A, color='black')
axes[0].set_title('AIDA-A')
axes[0].set_ylabel('Micro F1-Score')
axes[0].set_xticklabels(models, rotation=90, ha="center")
axes[0].set_ylim([0, max_f1_score])  # Set common y-axis limit
axes[0].grid(True, linestyle='--', color='black')

# AIDA-B
axes[1].bar(models, f1_scores_AIDA_B, color='black')
axes[1].set_title('AIDA-B')
axes[1].set_xticklabels(models, rotation=90, ha="center")
axes[1].grid(True, linestyle='--', color='black')

# AIDA-C
axes[2].bar(models, f1_scores_AIDA_C, color='black')
axes[2].set_title('AIDA-C')
axes[2].set_xticklabels(models, rotation=90, ha="center")
axes[2].grid(True, linestyle='--', color='black')

# Overall title and layout adjustments
#plt.suptitle('F1-Score Comparison Across Models for AIDA-A, AIDA-B, and AIDA-C')
plt.tight_layout()

# Display the graphs
plt.show()
