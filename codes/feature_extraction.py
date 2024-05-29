import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score,  accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from models import LSTMModel, CNN1D
import numpy as np

# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, directory, type_a = 'late'):
        if type_a == 'late':
            positive = 'FN'
            negative = 'F-'
        elif type_a == 'early':
            positive = 'N'
            negative = 'PR'
        keypoints = [
            "nose",
            "left wrist",
            "right elbow",
            "left elbow",
            "left ankle",
            "head bottom",
            "head top",
            "right wrist",
            "left hip",
            "right shoulder",
            "right hip",
            "left shoulder",
            "left knee",
            "right ear",
            "right ankle",
            "left ear",
            "right knee"
        ]


        all_data = []
        labels = []
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and (negative in filename or positive in filename):
                filepath = os.path.join(directory, filename)
                data = pd.read_csv(filepath, header=0)
                # Masking out the points where visible is False
                data.loc[~data['visible'], ['x', 'y']] = 0

                # Check for missing keypoints
                existing_keypoints = set(data['keypoint'])
                missing_keypoints = [kp for kp in keypoints if kp not in existing_keypoints]

                # Add missing keypoints with NaN values
                for keypoint in missing_keypoints:
                    new_row = pd.DataFrame({
                        'tracked_from_frame': [0],
                        'keypoint': keypoint,
                        'frame_index': [0],
                        'x': [0],
                        'y': [0],
                        'visible': [False]
                    })

                    data = pd.concat([data, new_row], ignore_index=True)

                data_order_x = data.pivot(index='keypoint', columns='frame_index', values='x').fillna(0)
                data_order_y = data.pivot(index='keypoint', columns='frame_index', values='y').fillna(0)
                combined_data = torch.cat((torch.tensor(data_order_x.values), torch.tensor(data_order_y.values)))
                all_data.append(combined_data)
                labels.append(int(negative in filename))

        # For now cropping the frames to the shortest one
        min_dim_size = min(tensor.size(1) for tensor in all_data)
        # all_data = [tensor[:, :min_dim_size, :] for tensor in all_data]

        # Split each tensor into chunks of size min_dim_size
        new_all_data = []
        new_labels = []
        for id, tensor in enumerate(all_data):
            for i in range(tensor.size(1) // min_dim_size):
                start_index = i * min_dim_size
                chunk = tensor[:, start_index:start_index + min_dim_size]
                new_all_data.append(chunk)
                new_labels.append(labels[id])

        self.features = new_all_data
        self.labels = torch.tensor(new_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), label

def create_dataloaders(dataset, batch_size=16, test_split=0.2):
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
def create_dataloaders_Kfold(train_indices, val_indices, dataset, batch_size=16):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.squeeze().float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        if (epoch) % 10 == 0:
            # Evaluate on test data
            model.eval()
            total = 0
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    predicted = model(inputs).squeeze()
                    total += labels.size(0)

                    if predicted.dim() == 0:
                        predicted = predicted.unsqueeze(0)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            # Convert lists to numpy arrays for sklearn functions
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)

            # Compute AUROC
            auroc = roc_auc_score(all_labels, all_predictions)

            # Compute AUPR
            precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
            aupr = auc(recall, precision)

            # Compute F1 score
            f1 = f1_score(all_labels, (all_predictions > 0.5).astype(float))

            # Compute Accuracy
            accuracy = accuracy_score(all_labels,  (all_predictions > 0.5).astype(float))

            print(f"Test Accuracy: {accuracy}% ", f"AUROC: {auroc}% ", f"AUPR: {aupr}% ", f"F1-score: {f1}% " )
    return accuracy, auroc, aupr, f1

def test_model(model, test_loader):

    # Evaluate on test data
    model.eval()
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            predicted = model(inputs).squeeze()
            total += labels.size(0)

            if predicted.dim() == 0:
                predicted = predicted.unsqueeze(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Convert lists to numpy arrays for sklearn functions
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_predictions)

    # Compute AUPR
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    aupr = auc(recall, precision)

    # Compute F1 score
    f1 = f1_score(all_labels, (all_predictions > 0.5).astype(float))

    # Compute Accuracy
    accuracy = accuracy_score(all_labels,  (all_predictions > 0.5).astype(float))

    print(f"Test Accuracy: {accuracy}% ", f"AUROC: {auroc}% ", f"AUPR: {aupr}% ", f"F1-score: {f1}% " )
    return accuracy, auroc, aupr, f1


def cross_validate(model, dataset, k_folds=5, epochs=10, seed=42):
    # Split the dataset into training and test sets
    train_val_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=seed,
                                                       stratify=dataset.labels)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=32, shuffle=False)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    acc_ls, auroc_ls, aupr_ls, f1_ls = [], [], [], []
    best_model = None
    best_val_auroc = 0
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_val_indices)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        train_loader, val_loader = create_dataloaders_Kfold(train_val_indices[train_indices],
                                                            train_val_indices[val_indices], dataset)

        acc, auroc, aupr, f1_score = train_model(model, train_loader, val_loader, epochs=epochs)
        acc_ls.append(acc)
        auroc_ls.append(auroc)
        aupr_ls.append(aupr)
        f1_ls.append(f1_score)

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

    # Train final model on the entire training + validation set

    test_acc, test_auroc, test_aupr, test_f1_score = test_model(best_model, test_loader)

    print('--------------------------------')
    print('TEST SET RESULTS')
    print('--------------------------------')
    print(f'Test accuracy: {test_acc}')
    print(f'Test AUROC: {test_auroc}')
    print(f'Test AUPR: {test_aupr}')
    print(f'Test F1 score: {test_f1_score}')

def cross_validate_SplitOnly(model, dataset, k_folds=5, epochs=10, seed = 42):
    kfold = KFold(n_splits=k_folds, shuffle=True,random_state=seed)
    acc_ls, auroc_ls, aupr_ls, f1_ls = [], [], [], []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        train_loader, val_loader = create_dataloaders_Kfold(train_indices, val_indices, dataset)
        acc, auroc, aupr, f1_score = train_model(model, train_loader, val_loader, epochs=epochs)
        acc_ls.append(acc)
        auroc_ls.append(auroc)
        aupr_ls.append(aupr)
        f1_ls.append(f1_score)


        # Optionally save the model after each fold
        # torch.save(model.state_dict(), f'model_fold_{fold}.pth')

    print('--------------------------------')
    print('K-FOLD CROSS VALIDATION RESULTS')
    print('--------------------------------')
    print(f'Average accuracy: {np.mean(acc_ls)}')
    print(f'Average AUROC: {np.mean(auroc_ls)}')
    print(f'Average AUPR: {np.mean(aupr_ls)}')
    print(f'Average F1 score: {np.mean(f1_ls)}')


# Input arguments
type_a = 'early'
seed = 42
epochs = 100
folds = 5

set_seeds(seed)
# Create dataset
directory = '../output/'
dataset = CustomDataset(directory,type_a)
if type_a == 'late':
    length = 467
elif type_a == 'early':
    length = 616
model = CNN1D(sequence_length=length)
# model = LSTMModel(input_size=length)

# Training one data split
# train_loader, test_loader = create_dataloaders(dataset)
# train_model(model, train_loader, test_loader, epochs=100)

# Cross validation
cross_validate(model, dataset, k_folds=folds, epochs=epochs, seed = seed)

pass