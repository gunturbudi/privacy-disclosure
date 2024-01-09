from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, SentenceTransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_fold(data_folder, fold):
    # Paths for the train and test files
    train_file = f'fasttext_train_fold_{fold}.txt'
    test_file = f'fasttext_test_fold_{fold}.txt'

    # 1. Load the corpus
    corpus: Corpus = ClassificationCorpus(data_folder,
                                          test_file=test_file,
                                          train_file=train_file,
                                          label_type='label')

    # 2. Create the label dictionary
    label_dict = corpus.make_label_dictionary(label_type='label')

    # 3. Make a list of sentence transformer models
    transformer_models = {
        'bert-base-nli-mean-tokens': 'bert-base-nli-mean-tokens',
        'roberta-base-nli-stsb-mean-tokens': 'roberta-base-nli-stsb-mean-tokens',
        'all-mpnet-base-v2': 'all-mpnet-base-v2',
    }

    for model_name, transformer_model in transformer_models.items():
        folder_name = f"{model_name}_fold_{fold}"
        output_disclosure_folder = "output_disclosure_sent"  # Subfolder for outputs
        output_folder = os.path.join(data_folder, output_disclosure_folder, folder_name)

        # Check if the model has already been trained
        model_path = os.path.join(output_folder, 'final-model.pt')
        if os.path.exists(model_path):
            print(f"Model already trained for fold {fold} with embeddings {model_name}. Skipping training.")
            continue

        # Check if the model has already been trained
        if os.path.exists(os.path.join(output_folder, 'final-model.pt')):
            continue

        print(f"Training fold {fold} with embeddings {model_name}")

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # 4. Initialize document embedding using DocumentCNNEmbeddings
        document_embeddings = SentenceTransformerDocumentEmbeddings(model=transformer_model)

        # 5. Create the text classifier
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type='label')

        # 6. Initialize the text classifier trainer
        trainer = ModelTrainer(classifier, corpus)

        # 7. Start the training
        trainer.train(output_folder,
                      learning_rate=0.1,
                      mini_batch_size=32,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=150)

def main():
    data_folder = 'data_priv'  # Update with the actual path

    for fold in range(5):
        train_fold(data_folder, fold)

main()
