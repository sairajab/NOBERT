from multiview_model import prepare_data, DEEPPLANT_MODEL_DIR
import numpy as np
import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from deepplant import build_model
import os
from tqdm import tqdm
import joblib
from sklearn.metrics import average_precision_score


def save_deepplant_embeddings(model, dataloader, device, output_file):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_input_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Saving embeddings"):
            input_ids, sequences, chromatin_features, labels = batch
            embeddings = model(sequences.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_input_ids.append(list(input_ids))

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_input_ids = np.concatenate(all_input_ids, axis=0)
    
    np.savez(output_file,
             embeddings=all_embeddings,
             labels=all_labels,
             input_ids=all_input_ids)



def load_and_concat(npz_path, chromatin_df):
    """Load embeddings and horizontally concatenate chromatin features."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']               
    labels     = data['labels']
    input_ids  = data['input_ids']                

    chrom = chromatin_df.loc[input_ids].values    # (N, D_chrom)

    X = chrom #np.concatenate([embeddings, chrom], axis=1)  # (N, D_emb + D_chrom)
    return X, labels, input_ids





if __name__ == "__main__":
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs_path = "/s/chromatin/a/nobackup/Saira/SATORI_ENHANCEMENTS/roy/enhancer_dataset_cleaned.csv"
    chromatin_path = "../DATA/enhancers/enh_600_2w_disc.csv"

    train_loader, val_loader, test_loader = prepare_data(seqs_path, chromatin_path, aggregation="concat")
    model =  build_model(new_model=True, model_path=os.path.join(DEEPPLANT_MODEL_DIR, 'model_AT_CSP.pt'), json_path=os.path.join(DEEPPLANT_MODEL_DIR, 'config_AT_2500.json'))

    model.to(device)
    print("Extracting and saving embeddings...")

    
    # save_deepplant_embeddings(model, train_loader, device, "train_embeddings.npz")
    # save_deepplant_embeddings(model, val_loader, device, "val_embeddings.npz")
    # save_deepplant_embeddings(model, test_loader, device, "test_embeddings.npz")


    # Load embeddings
    data = np.load("train_embeddings.npz", allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    input_ids = data['input_ids']
    X_train, y_train = embeddings, labels

    data = np.load("val_embeddings.npz", allow_pickle=True)
    X_val, y_val = data['embeddings'], data['labels']

    data = np.load("test_embeddings.npz", allow_pickle=True)
    X_test, y_test = data['embeddings'], data['labels']


    chromatin_data = pd.read_csv(chromatin_path)
    chromatin_data.set_index('actual', inplace=True)
    if 'window.1' in chromatin_data.columns:
        chromatin_data.drop(columns=['window.1'], inplace=True)
    if 'window' in chromatin_data.columns:
        chromatin_data.drop(columns=['window'], inplace=True)

    print(chromatin_data.head())

    

    #chromatin_data = (chromatin_data.groupby(chromatin_data.index).apply(lambda g: pd.Series(g.values.flatten())))

    # max aggregation

    chromatin_data = chromatin_data.groupby(chromatin_data.index).max()

    X_train, y_train, _ = load_and_concat("train_embeddings.npz", chromatin_data)
    X_val,   y_val,   _ = load_and_concat("val_embeddings.npz",   chromatin_data)
    X_test,  y_test,  _ = load_and_concat("test_embeddings.npz",  chromatin_data)

    print(X_train.shape, X_val.shape, X_test.shape)



    print(X_train.shape, X_val.shape, X_test.shape)

    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
    clf.fit(X_train, y_train)

    # save the model

    
    joblib.dump(clf, "svm_model_chrom.pkl")

    print("val accuracy: ", clf.score(X_val, y_val))
    print("test accuracy:", clf.score(X_test, y_test))

    y_val_scores  = clf.decision_function(X_val)
    y_test_scores = clf.decision_function(X_test)

    val_aupr  = average_precision_score(y_val,  y_val_scores)
    test_aupr = average_precision_score(y_test, y_test_scores)

    print(f"val AUPR:  {val_aupr:.4f}")
    print(f"test AUPR: {test_aupr:.4f}")

