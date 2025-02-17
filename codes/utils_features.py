import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
from collections import defaultdict
import numpy as np
import torch
import os
import wandb


def set_seeds(seed=42):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders_Kfold(train_indices, val_indices, dataset, batch_size=16):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, directory, type_a='late', feature_type='coordinates', label_method=None):
        """
        Args:
            directory (str): Directory where the tracked points are stored.
            type_a (str): Type of classification ('late' or 'early').
            feature_type (str): Type of features to use ('coordinates', 'angles', or 'both').
        """
        if type_a == 'late':
            positive = '_FN_c'
            negative = '_F-_c'
        elif type_a == 'early':
            positive = '_N_c'
            negative = '_PR_c'

        self.feature_type = feature_type

        if label_method is None:
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

        elif label_method == 'aggpose':
            keypoints = [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "joint_7",
                "joint_8",
                "joint_9",
                "joint_10",
                "joint_11",
                "joint_12",
                "joint_13",
                "joint_14",
                "joint_15",
                "joint_16",
                "joint_17",
                "joint_18",
                "joint_19",
                "joint_20"
            ]


        all_data = []
        labels = []
        original_indices = []  # To track the original sample index for each chunk
        seen_samples = set()  # To track seen samples and avoid duplicates of same subject
        group_identifiers = defaultdict(list) 
        for idx, filename in enumerate(os.listdir(directory)):
            if filename.endswith(".csv") and (negative in filename or positive in filename):

                # Extract the number and condition part (e.g., '57_PR') to check for duplicates of same subject id
                sample_id = '_'.join(filename.split('_')[4:6])
                # Skipping duplicated subjects
                if sample_id in seen_samples:
                    continue
                    # pass # If you do not care about duplicated subjects

                seen_samples.add(sample_id)

                filepath = os.path.join(directory, filename)
                data = pd.read_csv(filepath, header=0)

                # Masking out the points where visible is False
                data.loc[~data['visible'], ['x', 'y']] = 0

                # if label_method is None:
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

                combined_data = None

                if self.feature_type == 'coordinates' or self.feature_type == 'both':
                    # Include keypoint coordinates
                    coordinates_data = torch.cat((torch.tensor(data_order_x.values), torch.tensor(data_order_y.values)))
                    combined_data = coordinates_data

                if self.feature_type == 'angles' or self.feature_type == 'both':
                    # Compute angles between certain keypoints
                    angles = self.compute_angles(data_order_x, data_order_y, keypoints, label_method)
                    angles_tensor = torch.tensor(angles)  # Add a dimension to match combined_data
                    if combined_data is None:
                        combined_data = angles_tensor
                    else:
                        combined_data = torch.cat((combined_data, angles_tensor), dim=0)

                all_data.append(combined_data)
                labels.append(int(negative in filename))

        # For now cropping the frames to the shortest one
        self.min_dim_size = min(tensor.size(1) for tensor in all_data)

        # Split each tensor into chunks of size min_dim_size
        new_all_data = []
        new_labels = []
        for id, tensor in enumerate(all_data):
            for i in range(tensor.size(1) // self.min_dim_size):
                start_index = i * self.min_dim_size
                chunk = tensor[:, start_index:start_index + self.min_dim_size]
                new_all_data.append(chunk)
                new_labels.append(labels[id])

                # Track the original index for each chunk
                original_indices.append(id)

        self.features = new_all_data
        self.labels = torch.tensor(new_labels)
        self.original_indices = original_indices

    def compute_angles(self, data_order_x, data_order_y, keypoints, label_method=None):
        # Define pairs of keypoints for which angles will be calculated
        if label_method is None:
            angle_pairs = [
                ("head top", "nose", "head bottom"),  # head top to neck angle
                ("right ear", "nose", "left ear"),  # left to right ear angle
                ("left elbow", "left shoulder", "head bottom"),  # head to shoulder angle
                ("right elbow", "right shoulder", "head bottom"),  # head to shoulder angle
                ("left wrist", "left elbow", "left shoulder"),  # left elbow angle
                ("right wrist", "right elbow", "right shoulder"),  # right elbow angle
                ("left knee", "left hip", "left shoulder"),  # left hip angle
                ("right knee", "right hip", "right shoulder"),  # right hip angle
                ("left hip", "left knee", "left ankle"),  # left knee angle
                ("right hip", "right knee", "right ankle")  # right knee angle
            ]
        elif label_method == "aggpose":
            angle_pairs = [
                ("joint_19", "joint_18", "joint_0"),  # head top to neck angle
                ("joint_19", "joint_18", "joint_9"),  # left to right ear angle
                ("joint_2", "joint_1", "joint_19"),  # head to shoulder angle
                ("joint_11", "joint_10", "joint_19"),  # head to shoulder angle
                ("joint_3", "joint_2", "joint_1"),  # left elbow angle
                ("joint_12", "joint_11", "joint_10"),  # right elbow angle
                ("joint_15", "joint_14", "joint_10"),  # left hip angle
                ("joint_6", "joint_5", "joint_1"),  # right hip angle
                ("joint_14", "joint_15", "joint_16"),  # left knee angle
                ("joint_5", "joint_6", "joint_7")  # right knee angle
            ]

        angles = []

        for kp1, kp2, kp3 in angle_pairs:
            if kp1 in keypoints and kp2 in keypoints and kp3 in keypoints:
                p1_x, p1_y = data_order_x.loc[kp1], data_order_y.loc[kp1]
                p2_x, p2_y = data_order_x.loc[kp2], data_order_y.loc[kp2]
                p3_x, p3_y = data_order_x.loc[kp3], data_order_y.loc[kp3]

                # Calculate vectors
                v1 = np.array([p1_x - p2_x, p1_y - p2_y])  # vector from 'kp2' tp 'kp1'
                v2 = np.array([p3_x - p2_x, p3_y - p2_y])  # vector from 'kp2' to 'kp3'

                # Compute the norms of the vectors
                norm_v1 = np.linalg.norm(v1, axis=0)
                norm_v2 = np.linalg.norm(v2, axis=0)

                # Create a mask for where either norm is zero
                zero_norm_mask = (norm_v1 == 0) | (norm_v2 == 0)

                # Initialize an array for angles
                angle = np.zeros(p1_x.shape)

                # Compute angles only where norms are non-zero
                if not zero_norm_mask.all():  # If not all frames have zero norms
                    valid_dot_product = np.einsum('ij,ij->j', v1[:, ~zero_norm_mask], v2[:, ~zero_norm_mask])
                    valid_cosine_angle = valid_dot_product / (norm_v1[~zero_norm_mask] * norm_v2[~zero_norm_mask])
                    valid_angle = np.arccos(np.clip(valid_cosine_angle, -1.0, 1.0))

                    # Assign computed angles back to the array
                    angle[~zero_norm_mask] = valid_angle

                angles.append(angle)

        # Return angles as a numpy array (flattened)
        return np.stack(angles, axis=0) #np.concatenate(angles)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), label


def train_model(model, train_loader, test_loader, epochs=10, use_wandb=False, fold=None, wandb_config=None, lr_input=None):
    run_name = "" if fold is None else f"Fold_{fold + 1}"
    criterion = nn.BCELoss()

    if lr_input is not None:
        lr = lr_input
    else:
        lr = 0.00001 if wandb_config is None else wandb_config.learning_rate

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.squeeze().float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        if (epoch) % 10 == 0:
            # Evaluate on test data
            model.eval()
            total = 0
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    if torch.cuda.is_available():
                        inputs = inputs.to("cuda:0")

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
            accuracy = accuracy_score(all_labels, (all_predictions > 0.5).astype(float))

            print(f"Test Accuracy: {accuracy} ", f"AUROC: {auroc} ", f"AUPR: {aupr} ", f"F1-score: {f1} ")

            if use_wandb:
                wandb.log(data={f"Evaluation_train/{run_name}_Eval_Accuracy": accuracy,
                                f"Evaluation_train/{run_name}_Eval_AUROC": auroc,
                                f"Evaluation_train/{run_name}_Eval_AUPR": aupr,
                                f"Evaluation_train/{run_name}_Eval_F1-score": f1}, commit=False)

        if use_wandb:
            wandb.log(data={f"Train/{run_name}_Loss": running_loss / len(train_loader)}, commit=True)

    return accuracy, auroc, aupr, f1


def test_model(model, test_loader, model_type):
    if model_type == 'RF':
        val_pred = model.predict(test_loader[0])
        all_predictions = model.predict_proba(test_loader[0])[:, 1]
        all_labels = test_loader[1]
    else:
        # Evaluate on test data
        model.eval()
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda:0")

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
    accuracy = accuracy_score(all_labels, (all_predictions > 0.5).astype(float))

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, (all_predictions > 0.5).astype(float))

    print(f"Test Accuracy: {accuracy} ", f"AUROC: {auroc} ", f"AUPR: {aupr} ", f"F1-score: {f1} ")
    return accuracy, auroc, aupr, f1, cm


def cross_validate_SplitOnly(model, dataset, k_folds=5, epochs=10, seed=42):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
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
