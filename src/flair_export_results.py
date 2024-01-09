import os
import re
import pandas as pd

def parse_log_file(log_file_path):
    """
    Parse the training.log file to extract macro and weighted average metrics using regex.
    """
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Regex patterns to extract metrics
    overall_pattern = r"- F-score \(micro\) ([0-9.]+)\n- F-score \(macro\) ([0-9.]+)\n- Accuracy ([0-9.]+)"
    averages_pattern = r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\nweighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"

    # Extracting overall metrics
    overall_metrics = re.search(overall_pattern, log_content)
    averages_metrics = re.search(averages_pattern, log_content)

    if overall_metrics and averages_metrics:
        metrics = {
            'F-score (micro)': float(overall_metrics.group(1)),
            'F-score (macro)': float(overall_metrics.group(2)),
            'Accuracy': float(overall_metrics.group(3)),
            'Macro Avg Precision': float(averages_metrics.group(1)),
            'Macro Avg Recall': float(averages_metrics.group(2)),
            'Macro Avg F1-Score': float(averages_metrics.group(3)),
            'Weighted Avg Precision': float(averages_metrics.group(4)),
            'Weighted Avg Recall': float(averages_metrics.group(5)),
            'Weighted Avg F1-Score': float(averages_metrics.group(6)),
        }
        return metrics
    else:
        # Return None or a default value if no match is found
        return None

def main():
    data_folder = 'data_priv'  # Base data folder
    output_folders = ['output_disclosure_gru', 'output_disclosure_lstm', 'output_disclosure_sent', 'output_disclosure_cnn']  # Add other folders as needed
    results = []

    for output_folder in output_folders:
        # List all embedding subdirectories in the output folder
        embedding_folders = [f for f in os.listdir(os.path.join(data_folder, output_folder))
                             if os.path.isdir(os.path.join(data_folder, output_folder, f))]
        
        for embedding_folder in embedding_folders:
            # Extract fold number from the folder name
            fold_match = re.search(r'_fold_(\d+)', embedding_folder)
            if fold_match:
                fold_number = fold_match.group(1)
                log_file_path = os.path.join(data_folder, output_folder, embedding_folder, 'training.log')
                if os.path.exists(log_file_path):
                    metrics = parse_log_file(log_file_path)
                    if metrics:  # Check if metrics is not None
                        metrics['Fold'] = fold_number
                        metrics['Output Folder'] = output_folder
                        metrics['Embedding'] = embedding_folder.split('_fold_')[0]  # Extract embedding name
                        results.append(metrics)

    # Create DataFrame and reorder columns
    df = pd.DataFrame(results)
    column_order = ['Output Folder', 'Embedding', 'Fold', 'Accuracy', 'F-score (micro)', 'F-score (macro)',
                    'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1-Score',
                    'Weighted Avg Precision', 'Weighted Avg Recall', 'Weighted Avg F1-Score']
    df = df[column_order]

    df.to_excel('training_performance_comparison.xlsx', index=False)

if __name__ == "__main__":
    main()
