import os
import shutil
import torch.cuda
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from models import LSTMModel, CNN1D
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
from utils_features import train_model, test_model, set_seeds, create_dataloaders_Kfold, CustomDataset
import wandb
import time
import matplotlib.pyplot as plt


def cross_validate(model, dataset, args, model_type = 'CNN', k_folds=5, epochs=10, seed=42, batch_size=64, use_wandb=False, wandb_project=None,
                   wandb_run_prefix=None):
    # Group indices by original sample
    grouped_indices = {}
    for idx, original_idx in enumerate(dataset.original_indices):
        if original_idx not in grouped_indices:
            grouped_indices[original_idx] = []
        grouped_indices[original_idx].append(idx)

    # Flatten the grouped indices for splitting
    all_indices = list(grouped_indices.keys())

    # Splitting based on the number of original samples, not the number of chunks, to ensure no data leakage
    # Stratified split based on the original labels, every chunk of the same sample has the same label, therefore indices[0] is enough
    train_val_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=seed,
                                                       stratify=[dataset.labels[indices[0]] for indices in
                                                                 grouped_indices.values()])

    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    # Stratified K-Fold cross-validation on the training/validation set
    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    train_val_stratify_labels = [dataset.labels[grouped_indices[sample_id][0]] for sample_id in train_val_indices]

    # Initialize WandB runs if requested
    if use_wandb:
        run = wandb.init(project=wandb_project, name=f"{wandb_run_prefix}_{time.time()}", reinit=True)
        config_dict = {"epochs": epochs, "batch_size": batch_size, "seed": seed, "type_a": args.type_a, "model": model.__class__.__name__}

        if args.num_outlier_passes >= 0:
            config_dict["num_outlier_passes"] = args.num_outlier_passes

        wandb.config.update(config_dict)

    acc_ls, auroc_ls, aupr_ls, f1_ls = [], [], [], []
    best_model = None
    best_val_auroc = 0
    # for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(train_val_indices, train_val_stratify_labels)):

        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        # Map k-fold indices to the actual train/validation indices
        train_indices = [idx for i in train_idx for idx in grouped_indices[train_val_indices[i]]]
        val_indices = [idx for i in val_idx for idx in grouped_indices[train_val_indices[i]]]

        if model_type == 'RF':
            train_data = [dataset[i] for i in train_indices]
            val_data = [dataset[i] for i in val_indices]

            # Random forest process data
            train_X = np.array([x[0].numpy() for x in train_data])
            train_y = np.array([x[1].numpy() for x in train_data])
            val_X = np.array([x[0].numpy() for x in val_data])
            val_y = np.array([x[1].numpy() for x in val_data])
            # Flattening data TODO: Extract more informative features instead of flattening if desired
            train_X = train_X.reshape(train_X.shape[0], -1)
            val_X = val_X.reshape(val_X.shape[0], -1)

            # Scaling the data
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            val_X = scaler.transform(val_X)

            # Random forest train and evaluate
            model.fit(train_X, train_y)
            val_pred = model.predict(val_X)
            val_proba = model.predict_proba(val_X)[:, 1]

            acc = accuracy_score(val_y, val_pred)
            auroc = roc_auc_score(val_y, val_proba)
            precision, recall, _ = precision_recall_curve(val_y, val_proba)
            aupr = auc(recall, precision)
            f1 = f1_score(val_y, val_pred)

        elif model_type in ('CNN', 'LSTM'):
            train_loader, val_loader = create_dataloaders_Kfold(train_indices, val_indices, dataset, batch_size=batch_size)

            acc, auroc, aupr, f1 = train_model(model, train_loader, val_loader, epochs=epochs, use_wandb=use_wandb,
                                                     fold=fold)
        acc_ls.append(acc)
        auroc_ls.append(auroc)
        aupr_ls.append(aupr)
        f1_ls.append(f1)

        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_model = model

        # Optionally save the model after each fold
        # torch.save(model.state_dict(), f'model_fold_{fold}.pth')

    print('--------------------------------')
    print('K-FOLD CROSS VALIDATION RESULTS')
    print('--------------------------------')
    print(f'Average accuracy: {np.mean(acc_ls)}')
    print(f'Average AUROC: {np.mean(auroc_ls)}')
    print(f'Average AUPR: {np.mean(aupr_ls)}')
    print(f'Average F1 score: {np.mean(f1_ls)}')

    if use_wandb:
        wandb.log({"Evaluation/Average accuracy": np.mean(acc_ls), "Evaluation/Average AUROC": np.mean(auroc_ls),
                   "Evaluation/Average AUPR": np.mean(aupr_ls), "Evaluation/Average F1 score": np.mean(f1_ls)})

    if model_type == 'RF':
        test_indices = [idx for original_idx in test_indices for idx in grouped_indices[original_idx]]
        test_data = [dataset[i] for i in test_indices]
        test_X = np.array([x[0].numpy() for x in test_data])
        test_y = np.array([x[1].numpy() for x in test_data])
        test_X = test_X.reshape(test_X.shape[0], -1)
        test_X = scaler.transform(test_X)
        test_loader = [test_X, test_y]
    else:
        test_loader = DataLoader(
            Subset(dataset, [idx for original_idx in test_indices for idx in grouped_indices[original_idx]]),
            batch_size=batch_size,
            shuffle=False)

    # Train final model on the entire training + validation set
    test_acc, test_auroc, test_aupr, test_f1_score, cm = test_model(best_model, test_loader, model_type)

    # Create confusion matrix image
    cm_image = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_image.plot()

    print('--------------------------------')
    print('TEST SET RESULTS')
    print('--------------------------------')
    print(f'Test accuracy: {test_acc}')
    print(f'Test AUROC: {test_auroc}')
    print(f'Test AUPR: {test_aupr}')
    print(f'Test F1 score: {test_f1_score}')

    if use_wandb:
        wandb.log({"Test/Test accuracy": test_acc, "Test/Test AUROC": test_auroc, "Test/Test AUPR": test_aupr,
                   "Test/Test F1 score": test_f1_score,
                   "Test/Confusion Matrix": plt})
        run_dir = os.path.dirname(run.dir)
        run.finish()

        # Delete local wandb files
        shutil.rmtree(run_dir)


def main():
    parser = argparse.ArgumentParser(description='Run cross-validation for model training.')
    parser.add_argument('--seed', type=int, default=20, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--type_a', type=str, choices=['late', 'early'], default='late',
                        help='Type of classification problem')
    parser.add_argument('--directory', type=str, default='../output/tracked_merged',
                        help='Directory with the tracked coordinates')
    parser.add_argument('--model', type=str, default='CNN',
                        help='Model architecture to use for training. Choose between CNN, LSTM and RF.')
    parser.add_argument('--num_iterations', type=int, default=1, help='Number of iterations')
    parser.add_argument("--wandb", action='store_true', default=False, help="Use WandB")
    parser.add_argument("--wandb_project", type=str, default="GMA Project", help="WandB project name")
    parser.add_argument("--wandb_run_prefix", type=str, default="run", help="WandB run prefix")
    parser.add_argument("--num_outlier_passes", type=int, default=-1, help="Number of outlier passes")
    args = parser.parse_args()

    set_seeds(args.seed)

    # Initiate wandb if requested
    if args.wandb:
        wandb_URL = os.environ.get('WANDB_LOCAL_URL')
        wandb.login(host=wandb_URL)

    # Randomly generate a new seed for each iteration
    seeds = np.random.randint(0, 10000, args.num_iterations)

    # Do iterations
    for i in range(args.num_iterations):
        set_seeds(seeds[i])
        # Create dataset
        dataset = CustomDataset(args.directory, args.type_a)

        # Set model
        if args.model == 'CNN':
            model = CNN1D(sequence_length=dataset.min_dim_size)
        elif args.model == 'LSTM':
            model = LSTMModel(input_size=dataset.min_dim_size)
        elif args.model == 'RF':
            model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
        else:
            print("Model undefined")
            exit(-1)

        if torch.cuda.is_available() and args.model != "RF":
            model = model.to("cuda:0")

        # Training one data split
        # train_loader, test_loader = create_dataloaders(dataset)
        # train_model(model, train_loader, test_loader, epochs=100)

        # Cross validation
        cross_validate(model, dataset,args, model_type = args.model,  k_folds=args.folds, epochs=args.epochs, seed=seeds[i],
                       batch_size=args.batch_size,
                       use_wandb=args.wandb, wandb_project=args.wandb_project, wandb_run_prefix=args.wandb_run_prefix)


if __name__ == "__main__":
    main()
