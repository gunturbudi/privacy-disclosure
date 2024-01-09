import pandas as pd

def convert_to_fasttext(input_file_path, output_file_path):
    # Read the tab-separated file
    data = pd.read_csv(input_file_path, sep='\t',encoding="utf-8")

    # Open the output file
    with open(output_file_path, 'w', encoding="utf-8") as output_file:
        # Iterate over each row in the DataFrame
        for _, row in data.iterrows():
            # Write to the output file in FastText format
            line = f"__label__{row['label']} {row['story']}\n"
            output_file.write(line)

input_train_file = 'data_train_classification_tab.txt'  # Update with your train data file path
output_train_file = 'data_train_classification_fasttext.txt'  # Path for the output file in FastText format
convert_to_fasttext(input_train_file, output_train_file)

input_test_file = 'data_test_classification_tab.txt'    # Update with your test data file path
output_test_file = 'data_test_classification_fasttext.txt'   # Path for the output file in FastText format
convert_to_fasttext(input_test_file, output_test_file)

input_train_file = 'eda_data_train_classification_tab.txt'  # Update with your train data file path
output_train_file = 'eda_data_train_classification_fasttext.txt'  # Path for the output file in FastText format
convert_to_fasttext(input_train_file, output_train_file)