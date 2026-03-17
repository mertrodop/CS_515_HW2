import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(params: dict, train: bool = True) -> transforms.Compose:
    """Return a torchvision transform pipeline for the given dataset and split."""
    mean, std = params["mean"], params["std"]

    if params["dataset"] == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if params.get("resize_224", False):
            if train:
                return transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ])
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(params: dict) -> tuple[DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader) for the dataset specified in params."""
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float]:
    """Run one training epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on loader; return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def run_training(model: nn.Module, params: dict, device: torch.device) -> dict:
    """Train model using params; save best checkpoint; return history dict."""
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable,lr=params["learning_rate"],weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = None
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, params["log_interval"])
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
    return history


class DistillationLoss(nn.Module):
    """Hinton knowledge distillation loss.

    loss = alpha * T² * KL(student_soft || teacher_soft)
         + (1 - alpha) * CrossEntropy(student, hard_labels)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        soft_loss = self.kl(
            nn.functional.log_softmax(student_logits / self.T, dim=1),
            nn.functional.softmax(teacher_logits  / self.T, dim=1),
        ) * (self.T ** 2)
        hard_loss = self.ce(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def run_kd_training(
    student: nn.Module,
    teacher: nn.Module,
    params: dict,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict:
    """Train *student* using knowledge distillation from a frozen *teacher*."""
    train_loader, val_loader = get_loaders(params)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if criterion is None:
        criterion = DistillationLoss(
            temperature=params.get("kd_temperature", 4.0),
            alpha=params.get("kd_alpha", 0.7),
        )
    ce_eval = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = None
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, params["epochs"] + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            optimizer.zero_grad()
            student_logits = student(imgs)
            loss = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += student_logits.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

            if (batch_idx + 1) % params["log_interval"] == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}] "
                      f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

        tr_loss, tr_acc = total_loss / n, correct / n
        val_loss, val_acc = validate(student, val_loader, ce_eval, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch}/{params['epochs']}  "
              f"train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    student.load_state_dict(best_weights)
    print(f"\nKD training done. Best val accuracy: {best_acc:.4f}")
    return history


class TeacherGuidedLSLoss(nn.Module):
    """Dynamic label smoothing KD using only teacher's true-class probability.

    Effective smoothing epsilon is example-wise:
      epsilon_i = 1 - softmax(teacher_logits/T)[i, y_i]

    Soft label for sample i:
      - true class:    p = softmax(teacher_logits / T)[i, y_i]
      - other classes: (1 - p) / (C - 1)  each (uniform)
    Loss = cross-entropy(student_logits, soft_labels)

    temperature > 1 → teacher more uniform → more smoothing overall.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits, targets):
        B, C = student_logits.shape
        teacher_probs = nn.functional.softmax(teacher_logits / self.T, dim=1)
        p_true = teacher_probs[torch.arange(B, device=student_logits.device), targets]  # (B,)
        # build soft labels
        soft_labels = (1 - p_true).unsqueeze(1).expand(B, C) / (C - 1)
        soft_labels = soft_labels.clone()
        soft_labels[torch.arange(B, device=student_logits.device), targets] = p_true
        # cross-entropy with soft labels
        log_probs = nn.functional.log_softmax(student_logits, dim=1)
        return -(soft_labels * log_probs).sum(dim=1).mean()