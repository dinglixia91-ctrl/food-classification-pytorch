# ================================================
# Food Image Classification System with Multi-GPU Support
# ================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(6666)


# ================================================
# 1. Data Loading and Preprocessing
# ================================================
class FoodDataset(Dataset):
    def __init__(self, files, labels, transform=None, use_mixup=False, num_classes=11):
        self.files = files
        self.labels = labels
        self.transform = transform
        self.use_mixup = use_mixup
        self.num_classes = num_classes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.use_mixup and self.labels[idx] != -1:
            # Mixup data augmentation
            idx2 = random.randint(0, len(self) - 1)
            while self.labels[idx2] == -1:
                idx2 = random.randint(0, len(self) - 1)

            # Load images
            img1 = Image.open(self.files[idx]).convert('RGB')
            img2 = Image.open(self.files[idx2]).convert('RGB')

            # Apply transformations
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # Mixup coefficient
            lam = np.random.beta(0.4, 0.4)

            # Mix images and labels
            mixed_img = lam * img1 + (1 - lam) * img2

            # One-hot labels
            label1 = self.labels[idx]
            label2 = self.labels[idx2]

            # Store original labels for accuracy calculation
            original_labels = torch.tensor([label1, label2], dtype=torch.long)
            lam_values = torch.tensor([lam, 1 - lam])

            # For KLDivLoss, we need soft labels
            one_hot1 = torch.zeros(self.num_classes)
            one_hot1[label1] = 1
            one_hot2 = torch.zeros(self.num_classes)
            one_hot2[label2] = 1
            mixed_label = lam * one_hot1 + (1 - lam) * one_hot2

            return mixed_img, mixed_label, original_labels, lam_values
        else:
            # Normal loading
            img = Image.open(self.files[idx]).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                img = self.transform(img)

            return img, label


# Data transformations
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5584, 0.4528, 0.3458],
                         std=[0.2245, 0.2348, 0.2335])
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5584, 0.4528, 0.3458],
                         std=[0.2245, 0.2348, 0.2335])
])


def load_data(data_path):
    """Load data and return file paths and labels"""
    files = []
    labels = []

    for file in sorted(os.listdir(data_path)):
        if file.endswith('.jpg'):
            files.append(os.path.join(data_path, file))
            try:
                label = int(file.split('_')[0])
                labels.append(label)
            except:
                labels.append(-1)

    return files, labels


# ================================================
# 2. Model Definition with Multi-GPU Support
# ================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FoodClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(FoodClassifier, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Register Grad-CAM target layer
        self.target_layer = self.layer4[-1].conv2

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ================================================
# 3. Grad-CAM Visualization Class
# ================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()

        # Forward pass
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0

        logits.backward(gradient=one_hot, retain_graph=True)

        # Calculate weights
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Generate CAM
        cam = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.squeeze().cpu().numpy()


# ================================================
# 4. Multi-GPU Training Functions
# ================================================
def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False):
    """Train for one epoch with multi-GPU support"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        if use_mixup:
            images, soft_labels, orig_labels, lam = batch
            images = images.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)
            orig_labels = orig_labels.to(device, non_blocking=True)
            lam = lam.to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, soft_labels)

            # Calculate accuracy for mixup
            preds = logits.argmax(dim=1)

            # For mixup, we need to compare predictions with both original labels
            batch_correct = 0
            batch_total = 0
            for i in range(preds.size(0)):
                if preds[i] == orig_labels[i, 0] or preds[i] == orig_labels[i, 1]:
                    batch_correct += 1
                batch_total += 1

            correct += batch_correct
            total += batch_total
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Calculate accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()

        # Statistics
        total_loss += loss.item()

        # Update progress bar more frequently
        if batch_idx % 5 == 0:
            progress_bar.set_postfix({'Loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validation with multi-GPU support"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy, all_preds, all_labels


def train_fold(train_files, train_labels, val_files, val_labels, fold, num_epochs=40,
               batch_size=128, num_classes=11):
    """Train a single fold model with multi-GPU support"""
    print(f"\n{'=' * 50}")
    print(f"Training Fold {fold + 1}")
    print(f"{'=' * 50}")

    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Create datasets
    train_dataset = FoodDataset(train_files, train_labels, transform=train_tfm,
                                use_mixup=True, num_classes=num_classes)
    val_dataset = FoodDataset(val_files, val_labels, transform=test_tfm,
                              use_mixup=False, num_classes=num_classes)

    # Create data loaders with multi-GPU optimizations
    train_loader = DataLoader(train_dataset, batch_size=batch_size * n_gpus if n_gpus > 1 else batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * n_gpus if n_gpus > 1 else batch_size,
                            shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialize model, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoodClassifier(num_classes=num_classes)

    # Use DataParallel for multi-GPU training
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Use KLDivLoss for mixup training, CrossEntropyLoss for validation
    criterion_train = nn.KLDivLoss(reduction='batchmean')
    criterion_val = nn.CrossEntropyLoss()

    # Adjust learning rate based on batch size
    base_lr = 1e-4
    effective_batch_size = batch_size * n_gpus if n_gpus > 1 else batch_size
    lr = base_lr * (effective_batch_size / 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 使用CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_acc = 0
    best_model_state = None
    patience_counter = 0
    patience_limit = 20

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion_train,
                                            optimizer, device, use_mixup=True)

        # Validation
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion_val, device)

        # Update learning rate
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {current_lr:.6f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            # Save model state (handling DataParallel)
            if isinstance(model, nn.DataParallel):
                best_model_state = model.module.state_dict().copy()
            else:
                best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✅ New best accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⏹️ Early stopping triggered")
                break

    # Load best model
    if best_model_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    return model, history, best_acc


# ================================================
# 5. K-Fold Cross Validation Main Function
# ================================================
def k_fold_cross_validation(files, labels, n_splits=5, num_epochs=40, num_classes=11):
    """Perform K-fold cross validation with multi-GPU support"""
    print(f"\n{'=' * 60}")
    print(f"Starting {n_splits}-Fold Cross Validation")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"{'=' * 60}")

    # Create StratifiedKFold (maintain class distribution)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []
    histories = []
    fold_accuracies = []

    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
        # Get data for current fold
        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Train current fold
        model, history, best_acc = train_fold(
            train_files, train_labels, val_files, val_labels,
            fold, num_epochs, num_classes=num_classes
        )

        # Save results
        models.append(model)
        histories.append(history)
        fold_accuracies.append(best_acc)

        # Save model (handle DataParallel)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), f'model_fold_{fold + 1}.pth')
        print(f"✅ Fold {fold + 1} model saved, accuracy: {best_acc:.4f}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print overall results
    print(f"\n{'=' * 60}")
    print(f"K-Fold Cross Validation Completed")
    print(f"{'=' * 60}")
    print(f"Fold Accuracies: {fold_accuracies}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Best Accuracy: {np.max(fold_accuracies):.4f} (Fold {np.argmax(fold_accuracies) + 1})")

    return models, histories, fold_accuracies


# ================================================
# 6. Improved Model Ensemble and Prediction
# ================================================
class ImprovedEnsembleModel:
    """Improved ensemble model with weighted voting and better strategies"""

    def __init__(self, models, device=None):
        self.models = models
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move all models to device
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict(self, dataloader, method='weighted_soft_voting', weights=None):
        """Prediction methods: soft_voting, hard_voting, weighted_soft_voting, weighted_averaging"""
        all_predictions = []

        # If weights not provided, use equal weights
        if weights is None:
            weights = [1.0] * len(self.models)
        weights = torch.tensor(weights).to(self.device)
        weights = weights / weights.sum()  # Normalize weights

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc='Ensemble Prediction'):
                images = images.to(self.device, non_blocking=True)

                if method == 'soft_voting':
                    # Standard soft voting
                    batch_probs = []
                    for model in self.models:
                        logits = model(images)
                        probs = F.softmax(logits, dim=1)
                        batch_probs.append(probs)

                    avg_probs = torch.stack(batch_probs).mean(dim=0)
                    preds = avg_probs.argmax(dim=1)

                elif method == 'hard_voting':
                    # Standard hard voting
                    batch_preds = []
                    for model in self.models:
                        logits = model(images)
                        pred = logits.argmax(dim=1)
                        batch_preds.append(pred)

                    # Stack and take mode (most frequent)
                    batch_preds = torch.stack(batch_preds)
                    preds = torch.mode(batch_preds, dim=0)[0]

                elif method == 'weighted_soft_voting':
                    # Weighted soft voting
                    weighted_probs = []
                    for model, weight in zip(self.models, weights):
                        logits = model(images)
                        probs = F.softmax(logits, dim=1)
                        weighted_probs.append(probs * weight)

                    avg_probs = torch.stack(weighted_probs).sum(dim=0)
                    preds = avg_probs.argmax(dim=1)

                elif method == 'weighted_averaging':
                    # Weighted averaging
                    weighted_logits = []
                    for model, weight in zip(self.models, weights):
                        logits = model(images)
                        weighted_logits.append(logits * weight)

                    avg_logits = torch.stack(weighted_logits).sum(dim=0)
                    preds = avg_logits.argmax(dim=1)

                else:
                    raise ValueError(f"Unknown method: {method}")

                all_predictions.append(preds.cpu())

        return torch.cat(all_predictions)

    def predict_with_uncertainty(self, dataloader):
        """Predict with uncertainty estimation"""
        all_predictions = []
        all_uncertainties = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc='Ensemble Prediction with Uncertainty'):
                images = images.to(self.device, non_blocking=True)

                # Get predictions from all models
                all_model_probs = []
                for model in self.models:
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                    all_model_probs.append(probs)

                # Stack probabilities
                all_model_probs = torch.stack(all_model_probs)  # [n_models, batch_size, n_classes]

                # Compute ensemble prediction (average probability)
                avg_probs = all_model_probs.mean(dim=0)
                preds = avg_probs.argmax(dim=1)

                # Compute uncertainty as variance across models
                uncertainties = all_model_probs.var(dim=0).mean(dim=1)  # Variance across models, averaged over classes

                all_predictions.append(preds.cpu())
                all_uncertainties.append(uncertainties.cpu())

        predictions = torch.cat(all_predictions)
        uncertainties = torch.cat(all_uncertainties)

        return predictions, uncertainties

    def predict_proba(self, dataloader):
        """Predict probabilities (average of all models)"""
        all_probs = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc='Ensemble Probability Prediction'):
                images = images.to(self.device, non_blocking=True)
                batch_probs = []

                for model in self.models:
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                    batch_probs.append(probs)

                avg_probs = torch.stack(batch_probs).mean(dim=0)
                all_probs.append(avg_probs.cpu())

        return torch.cat(all_probs)


# ================================================
# 7. Visualization Functions (Fixed)
# ================================================
def plot_training_history(histories, save_path='training_history.png'):
    """Plot training history (average across all folds)"""
    n_folds = len(histories)
    n_epochs = len(histories[0]['train_loss'])

    # Calculate average history
    avg_history = {
        'train_loss': np.zeros(n_epochs),
        'train_acc': np.zeros(n_epochs),
        'val_loss': np.zeros(n_epochs),
        'val_acc': np.zeros(n_epochs)
    }

    for key in avg_history.keys():
        for i in range(n_folds):
            avg_history[key] += np.array(histories[i][key][:n_epochs])
        avg_history[key] /= n_folds

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curve
    axes[0].plot(avg_history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(avg_history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(avg_history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(avg_history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return cm


def plot_data_augmentation_examples(image_path, save_path='data_augmentation.png'):
    """Show data augmentation effects"""
    original_img = Image.open(image_path).convert('RGB')

    # Define different augmentation transformations
    augmentations = [
        ("Original", transforms.ToTensor()),
        ("Random Crop", transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
            transforms.ToTensor()
        ])),
        ("Horizontal Flip", transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])),
        ("Vertical Flip", transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])),
        ("Color Jitter", transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])),
        ("Random Rotation", transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])),
        ("Combined Augmentation", transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ]))
    ]

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (title, transform) in enumerate(augmentations):
        if idx >= len(axes):
            break

        img = transform(original_img)
        img = img.permute(1, 2, 0).numpy()

        # Denormalize (if needed)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')

    # Hide extra subplots
    for idx in range(len(augmentations), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def plot_grad_cam_visualizations(model, dataset, sample_indices, class_names, save_path='grad_cam.png'):
    """Plot Grad-CAM visualizations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle DataParallel model
    if isinstance(model, nn.DataParallel):
        target_layer = model.module.target_layer
    else:
        target_layer = model.target_layer

    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)

    # Denormalize
    inv_normalize = transforms.Normalize(
        mean=[-0.5584 / 0.2245, -0.4528 / 0.2348, -0.3458 / 0.2335],
        std=[1 / 0.2245, 1 / 0.2348, 1 / 0.2335]
    )

    # Create figure
    n_samples = min(len(sample_indices), 6)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(sample_indices[:n_samples]):
        img, true_label = dataset[idx]

        # Prediction
        input_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred_label = logits.argmax().item()

        # Generate CAM
        cam = gradcam(input_tensor, class_idx=pred_label)

        # Prepare images
        original_img = inv_normalize(img).cpu().permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)

        # Create heatmap
        heatmap = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay images
        overlay = 0.5 * original_img + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        # Plot
        axes[row, 0].imshow(original_img)
        axes[row, 0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(heatmap, cmap='jet')
        axes[row, 1].set_title(f"Grad-CAM Heatmap", fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(heatmap_colored)
        axes[row, 2].set_title(f"Colored Heatmap", fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title(f"Overlay\nPredicted: {class_names[pred_label]}", fontsize=10)
        axes[row, 3].axis('off')

    plt.suptitle('Grad-CAM Visualizations - Model Attention Areas', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def plot_model_comparison(baseline_acc, improved_acc, save_path='model_comparison.png'):
    """Plot model performance comparison"""
    models = ['Baseline Model (VGG-style)', 'Improved Model (ResNet-style)']
    accuracies = [baseline_acc, improved_acc]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, accuracies, color=['skyblue', 'lightcoral'])

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_ylim([0, 1])

    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def plot_fold_performance(fold_accuracies, save_path='fold_performance.png'):
    """Plot performance across different folds"""
    fig, ax = plt.subplots(figsize=(10, 6))

    folds = range(1, len(fold_accuracies) + 1)
    bars = ax.bar(folds, fold_accuracies, color='lightgreen')

    # Add average line
    avg_accuracy = np.mean(fold_accuracies)
    ax.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_accuracy:.4f}')

    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('K-Fold Cross Validation Performance', fontsize=14)
    ax.set_xticks(folds)
    ax.set_ylim([0, 1])

    # Add values on bars
    for bar, acc in zip(bars, fold_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def plot_ensemble_comparison(ensemble_results, save_path='ensemble_comparison.png'):
    """Plot ensemble strategies comparison"""
    strategies = list(ensemble_results.keys())
    accuracies = list(ensemble_results.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars = ax.bar(strategies, accuracies, color=colors[:len(strategies)])

    ax.set_xlabel('Ensemble Strategy', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ensemble Strategies Comparison', fontsize=14)
    ax.set_ylim([0, 1])

    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def plot_individual_vs_ensemble(individual_accs, ensemble_accs_dict, save_path='individual_vs_ensemble.png'):
    """Plot individual model vs ensemble model performance"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 获取集成策略的名称和准确率
    if isinstance(ensemble_accs_dict, dict):
        ensemble_names = list(ensemble_accs_dict.keys())
        ensemble_accs = list(ensemble_accs_dict.values())
    else:
        # 如果传入的是列表而不是字典
        ensemble_accs = ensemble_accs_dict
        ensemble_names = [f'Strategy {i + 1}' for i in range(len(ensemble_accs))]

    # Individual model accuracies
    M = len(individual_accs)
    N = len(ensemble_accs)

    # x轴位置
    x = np.arange(M + N)
    width = 0.6

    # 创建条形图
    bars = ax.bar(x, individual_accs + ensemble_accs, width, color='lightblue')

    # 设置不同部分的颜色
    for i in range(M):
        bars[i].set_color('skyblue')
    for i in range(M, M + N):
        bars[i].set_color('lightcoral')

    # 添加平均线
    avg_individual = np.mean(individual_accs)
    avg_ensemble = np.mean(ensemble_accs)
    ax.axhline(y=avg_individual, color='blue', linestyle='--', alpha=0.7,
               label=f'Avg Individual: {avg_individual:.4f}')
    ax.axhline(y=avg_ensemble, color='red', linestyle='--', alpha=0.7,
               label=f'Avg Ensemble: {avg_ensemble:.4f}')

    # 设置x轴标签
    x_labels = [f'Model {i + 1}' for i in range(M)] + ensemble_names
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Individual Models vs Ensemble Strategies Performance', fontsize=14)
    ax.set_ylim([0, 1.05])

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, individual_accs + ensemble_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9, rotation=0)

    # 添加分隔线
    if M > 0 and N > 0:
        ax.axvline(x=M - 0.5, color='black', linestyle='-', alpha=0.3, linewidth=2)
        # 添加文本标签
        ax.text(M / 2 - 0.5, 1.02, 'Individual Models', ha='center', fontsize=12, fontweight='bold')
        ax.text(M + N / 2 - 0.5, 1.02, 'Ensemble Strategies', ha='center', fontsize=12, fontweight='bold')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


# ================================================
# 8. Main Execution Pipeline with Multi-GPU Support
# ================================================
def main():
    print("Food Image Classification with K-Fold CV and Improved Model Ensemble")
    print("=" * 60)

    # Configuration parameters optimized for multi-GPU
    config = {
        'data_path': '/kaggle/input/ml2022spring-hw3b/food11/training',
        'valid_path': '/kaggle/input/ml2022spring-hw3b/food11/validation',
        'test_path': '/kaggle/input/ml2022spring-hw3b/food11/test',
        'n_splits': 5,
        'num_epochs': 40,
        'batch_size': 64,
        'num_classes': 11,
        'class_names': [f'class{i}' for i in range(11)]
    }

    # Display GPU information
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f" GPU {i}: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.2f} GB")

    # 1. Load data
    print("\n1. Loading data...")
    train_files, train_labels = load_data(config['data_path'])
    valid_files, valid_labels = load_data(config['valid_path'])

    print(f"Training set size: {len(train_files)}")
    print(f"Validation set size: {len(valid_files)}")

    # Check class distribution
    if len(train_labels) > 0:
        print("\nClass distribution in training set:")
        class_counts = {}
        for label in train_labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        for i in range(config['num_classes']):
            count = class_counts.get(i, 0)
            print(f" Class {i}: {count} samples")

    # 2. K-fold cross validation training with multi-GPU
    print("\n2. Starting K-fold cross validation training...")
    models, histories, fold_accuracies = k_fold_cross_validation(
        train_files, train_labels,
        n_splits=config['n_splits'],
        num_epochs=config['num_epochs'],
        num_classes=config['num_classes']
    )

    # 3. Evaluate individual models on validation set
    print("\n3. Evaluating individual models on validation set...")
    valid_dataset = FoodDataset(valid_files, valid_labels, transform=test_tfm,
                                use_mixup=False, num_classes=config['num_classes'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'] * max(1, n_gpus),
                              shuffle=False, num_workers=4, pin_memory=True)

    individual_accuracies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, model in enumerate(models):
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f'Evaluating Model {i + 1}'):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                preds = logits.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = (np.array(all_predictions) == np.array(all_labels)).mean()
        individual_accuracies.append(acc)
        print(f"Model {i + 1} accuracy on validation set: {acc:.4f}")

    print(f"\nIndividual model accuracies on validation set: {individual_accuracies}")
    print(f"Mean individual accuracy: {np.mean(individual_accuracies):.4f} ± {np.std(individual_accuracies):.4f}")

    # 4. Create improved ensemble model with weighting
    print("\n4. Creating improved ensemble model...")

    # Use fold accuracies as weights (models that performed better on their validation sets get higher weights)
    weights = [acc for acc in fold_accuracies]
    print(f"Weights based on fold accuracies: {weights}")

    ensemble = ImprovedEnsembleModel(models)

    # 5. Evaluate improved ensemble model on validation set
    print("\n5. Evaluating improved ensemble model on validation set...")

    # 只使用4个核心集成策略
    strategies = ['soft_voting', 'hard_voting', 'weighted_soft_voting', 'weighted_averaging']
    ensemble_results = {}

    for strategy in strategies:
        print(f"\nUsing {strategy} strategy...")

        if 'weighted' in strategy:
            predictions = ensemble.predict(valid_loader, method=strategy, weights=weights)
        else:
            predictions = ensemble.predict(valid_loader, method=strategy)

        accuracy = (predictions.numpy() == np.array(valid_labels)).mean()
        ensemble_results[strategy] = accuracy
        print(f"Accuracy: {accuracy:.4f}")

    # 6. Visualizations
    print("\n6. Generating visualizations...")

    # Create output directory
    os.makedirs('visualizations', exist_ok=True)

    try:
        # Training history
        print("Plotting training history...")
        history_fig = plot_training_history(histories, 'visualizations/training_history.png')
    except Exception as e:
        print(f"Error plotting training history: {e}")

    try:
        # Fold performance
        print("Plotting fold performance...")
        fold_fig = plot_fold_performance(fold_accuracies, 'visualizations/fold_performance.png')
    except Exception as e:
        print(f"Error plotting fold performance: {e}")

    try:
        # Individual vs ensemble comparison
        print("Plotting individual vs ensemble comparison...")
        individual_vs_ensemble_fig = plot_individual_vs_ensemble(
            individual_accuracies,
            ensemble_results,
            'visualizations/individual_vs_ensemble.png'
        )
    except Exception as e:
        print(f"Error plotting individual vs ensemble: {e}")

    try:
        # Ensemble comparison
        print("Plotting ensemble comparison...")
        ensemble_fig = plot_ensemble_comparison(ensemble_results, 'visualizations/ensemble_comparison.png')
    except Exception as e:
        print(f"Error plotting ensemble comparison: {e}")

    try:
        # Data augmentation examples
        if len(train_files) > 0:
            print("Plotting data augmentation examples...")
            sample_image = train_files[0]
            augmentation_fig = plot_data_augmentation_examples(sample_image, 'visualizations/data_augmentation.png')
    except Exception as e:
        print(f"Error plotting data augmentation: {e}")

    # Confusion matrix (using best ensemble strategy)
    best_strategy = max(ensemble_results, key=ensemble_results.get)
    print(f"\nBest ensemble strategy: {best_strategy} (Accuracy: {ensemble_results[best_strategy]:.4f})")

    try:
        # Recalculate predictions with best strategy
        if 'weighted' in best_strategy:
            predictions = ensemble.predict(valid_loader, method=best_strategy, weights=weights)
        else:
            predictions = ensemble.predict(valid_loader, method=best_strategy)

        print("Plotting confusion matrix...")
        cm_fig = plot_confusion_matrix(valid_labels, predictions.numpy(),
                                       config['class_names'], 'visualizations/confusion_matrix.png')
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    try:
        # Grad-CAM visualizations (using best single model)
        if len(models) > 0:
            print("Plotting Grad-CAM visualizations...")
            # Find best single model
            best_model_idx = np.argmax(individual_accuracies)
            if isinstance(models[best_model_idx], nn.DataParallel):
                grad_cam_model = models[best_model_idx].module
            else:
                grad_cam_model = models[best_model_idx]

            sample_indices = random.sample(range(len(valid_dataset)), min(6, len(valid_dataset)))
            grad_cam_fig = plot_grad_cam_visualizations(
                grad_cam_model, valid_dataset, sample_indices,
                config['class_names'], 'visualizations/grad_cam.png'
            )
    except Exception as e:
        print(f"Error plotting Grad-CAM: {e}")

    try:
        # Model comparison (assuming baseline accuracy is 0.543)
        baseline_acc = 0.543
        improved_acc = np.mean(individual_accuracies)  # Use individual model accuracies on validation set
        print("Plotting model comparison...")
        comparison_fig = plot_model_comparison(baseline_acc, improved_acc, 'visualizations/model_comparison.png')
    except Exception as e:
        print(f"Error plotting model comparison: {e}")

    # 7. Save results
    print("\n7. Saving results...")
    results = {
        'fold_accuracies': [float(acc) for acc in fold_accuracies],
        'individual_accuracies_on_valid': [float(acc) for acc in individual_accuracies],
        'mean_individual_accuracy': float(np.mean(individual_accuracies)),
        'std_individual_accuracy': float(np.std(individual_accuracies)),
        'ensemble_results': {k: float(v) for k, v in ensemble_results.items()},
        'best_ensemble_strategy': best_strategy,
        'best_ensemble_accuracy': float(ensemble_results[best_strategy]),
        'improvement_over_best_individual': float(ensemble_results[best_strategy] - np.max(individual_accuracies)),
        'improvement_over_mean_individual': float(ensemble_results[best_strategy] - np.mean(individual_accuracies)),
        'gpu_count': n_gpus,
        'num_epochs_per_fold': config['num_epochs']
    }

    # Save results to file
    import json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Number of GPUs used: {n_gpus}")
    print(
        f"Single model average accuracy on validation set: {results['mean_individual_accuracy']:.4f} ± {results['std_individual_accuracy']:.4f}")
    print(f"Best single model accuracy: {np.max(individual_accuracies):.4f}")
    print(f"Ensemble model best accuracy: {results['best_ensemble_accuracy']:.4f}")
    print(f"Improvement over best single model: {results['improvement_over_best_individual']:.4f}")
    print(f"Improvement over average single model: {results['improvement_over_mean_individual']:.4f}")
    print(f"Improvement over baseline: {results['best_ensemble_accuracy'] - baseline_acc:.4f}")
    print("=" * 60)

    # 8. Test set prediction (if test set exists)
    if os.path.exists(config['test_path']):
        print("\n8. Making predictions on test set...")
        test_files, _ = load_data(config['test_path'])
        test_dataset = FoodDataset(test_files, [-1] * len(test_files),
                                   transform=test_tfm, use_mixup=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'] * max(1, n_gpus),
                                 shuffle=False, num_workers=4, pin_memory=True)

        # Predict with ensemble model using best strategy
        if 'weighted' in best_strategy:
            test_predictions = ensemble.predict(test_loader, method=best_strategy, weights=weights)
        else:
            test_predictions = ensemble.predict(test_loader, method=best_strategy)

        # Save predictions
        test_results = []
        for file, pred in zip(test_files, test_predictions.numpy()):
            filename = os.path.basename(file)
            test_results.append({'filename': filename, 'prediction': int(pred)})

        # Save as CSV
        df = pd.DataFrame(test_results)
        df.to_csv('test_predictions.csv', index=False)
        print(f"Test set predictions saved to test_predictions.csv")
        print(f"Total test samples predicted: {len(test_results)}")

    return models, ensemble, results


# ================================================
# 9. Execute Main Function
# ================================================
if __name__ == "__main__":
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {n_gpus}")
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f" GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")
    else:
        print("No CUDA devices available. Using CPU.")

    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run main program
    try:
        print("\n" + "=" * 60)
        print("Starting training with multi-GPU support...")
        print("=" * 60)

        import time

        start_time = time.time()

        models, ensemble, results = main()

        end_time = time.time()
        training_time = (end_time - start_time) / 60
        print(f"\nTotal training time: {training_time:.2f} minutes")

        print("\n✅ All tasks completed!")
        print("📊 Generated visualizations (in 'visualizations/' folder):")
        visualizations = [
            "training_history.png",
            "fold_performance.png",
            "individual_vs_ensemble.png",
            "ensemble_comparison.png",
            "data_augmentation.png",
            "confusion_matrix.png",
            "grad_cam.png",
            "model_comparison.png"
        ]

        for viz in visualizations:
            if os.path.exists(f"visualizations/{viz}"):
                print(f" ✓ {viz}")
            else:
                print(f" ✗ {viz} (not generated)")

        print("📁 Saved files:")
        saved_files = [
                          "results.json",
                          "test_predictions.csv"
                      ] + [f"model_fold_{i}.pth" for i in range(1, 6)]

        for file in saved_files:
            if os.path.exists(file):
                print(f" ✓ {file}")
            else:
                print(f" ✗ {file} (not found)")

        # Display final results
        if 'results' in locals():
            print(f"\n📈 Final Results:")
            print(f" Baseline accuracy: 0.543")
            print(f" Best single model accuracy: {np.max(results['individual_accuracies_on_valid']):.4f}")
            print(f" Mean single model accuracy: {results['mean_individual_accuracy']:.4f}")
            print(f" Best ensemble accuracy: {results['best_ensemble_accuracy']:.4f}")
            print(f" Ensemble improvement over best single: {results['improvement_over_best_individual']:.4f}")
            print(f" Total improvement over baseline: {results['best_ensemble_accuracy'] - 0.543:.4f}")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()

        # Clear GPU cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()