import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Replace 'your_file.csv' with the path to your file
data = pd.read_csv('all_data_tab.txt', sep='\t', encoding="utf-8")


skf = StratifiedKFold(n_splits=5)

for fold, (train_index, test_index) in enumerate(skf.split(data, data['label'])):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    train_data.to_csv(f'train_fold_{fold}.csv', index=False)
    test_data.to_csv(f'test_fold_{fold}.csv', index=False)

    with open(f'fasttext_train_fold_{fold}.txt', 'w', encoding="utf-8") as train_file:
        for _, row in train_data.iterrows():
            train_file.write(f'__label__{row["label"]} {row["story"]}\n')

    with open(f'fasttext_test_fold_{fold}.txt', 'w', encoding="utf-8") as test_file:
        for _, row in test_data.iterrows():
            test_file.write(f'__label__{row["label"]} {row["story"]}\n')
