"""Model architecture and loss functions."""

import torch
from torch import nn
from torch.nn import functional
from transformers import AutoModel


class MaskedFocalLoss(nn.Module):
    """
    Focal Loss with masking for token classification.

    Focal Loss helps with class imbalance by down-weighting easy examples
    and focusing on hard examples. This is critical for PII detection where:
    - Most tokens are "O" (non-PII) - easy negative examples
    - Few tokens are actual PII - hard positive examples

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this is equivalent to standard cross-entropy.
    When gamma>0, it reduces the loss for well-classified examples.
    """

    def __init__(
        self,
        pad_label: int = -100,
        gamma: float = 2.0,
        alpha: float | None = None,
        class_weights: dict[int, float] | None = None,
        num_classes: int | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize the masked focal loss function.

        Args:
            pad_label: Label value for padding tokens (HuggingFace standard: -100)
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   gamma=0 is standard CE, gamma=2 is typical for detection tasks.
            alpha: Balance factor for positive/negative classes (optional)
            class_weights: Dictionary mapping class IDs to weights
            num_classes: Total number of classes
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.pad_label = pad_label
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights or {}
        self.num_classes = num_classes
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if self.num_classes is not None:
            self._build_weight_tensor()

    def _build_weight_tensor(self):
        """Build a weight tensor from class weights dictionary."""
        weight_tensor = torch.ones(self.num_classes, dtype=torch.float32)

        # Apply class weights - upweight PII classes (non-zero labels)
        for class_id, weight in self.class_weights.items():
            if 0 <= class_id < self.num_classes:
                weight_tensor[class_id] = float(weight)

        # If no explicit weights provided, auto-balance: upweight non-O classes
        if not self.class_weights and self.num_classes is not None:
            # O class (id=0) gets weight 1.0, PII classes get weight 5.0
            for i in range(1, self.num_classes):
                weight_tensor[i] = 5.0

        self.register_buffer("weight_tensor", weight_tensor)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the masked focal loss.

        Args:
            y_pred: Model predictions (logits) of shape (batch_size, seq_len, num_classes)
            y_true: True labels of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Computed loss value
        """
        # Create mask for non-padded elements (label != -100)
        label_mask = y_true != self.pad_label

        # Combine with attention mask if provided
        if attention_mask is not None:
            combined_mask = label_mask & (attention_mask.bool())
        else:
            combined_mask = label_mask

        # Create safe version of y_true to avoid errors with negative labels
        y_true_safe = torch.where(combined_mask, y_true, torch.zeros_like(y_true))

        # Compute softmax probabilities
        probs = functional.softmax(y_pred, dim=-1)

        # Get the probability of the true class for each token
        # Shape: (batch_size, seq_len)
        batch_size, seq_len, num_classes = y_pred.shape

        # Gather probabilities of true classes
        y_true_expanded = y_true_safe.unsqueeze(-1)  # (batch_size, seq_len, 1)
        p_t = probs.gather(dim=-1, index=y_true_expanded).squeeze(
            -1
        )  # (batch_size, seq_len)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smooth_loss = -torch.log(probs + 1e-8).mean(dim=-1)
            ce_loss = (
                1 - self.label_smoothing
            ) * ce_loss + self.label_smoothing * smooth_loss

        # Focal loss = focal_weight * ce_loss
        loss = focal_weight * ce_loss

        # Apply class weights if available
        if hasattr(self, "weight_tensor"):
            weight_tensor = self.weight_tensor.to(y_true_safe.device)
            sample_weights = weight_tensor[y_true_safe]
            loss = loss * sample_weights

        # Apply alpha balancing if specified
        if self.alpha is not None:
            # alpha for positive class, (1-alpha) for negative
            alpha_t = torch.where(
                y_true_safe > 0,
                torch.full_like(loss, self.alpha),
                torch.full_like(loss, 1 - self.alpha),
            )
            loss = alpha_t * loss

        # Apply combined mask (padding + attention)
        loss = torch.where(combined_mask, loss, torch.zeros_like(loss))

        # Apply reduction
        if self.reduction == "mean":
            total_loss = torch.sum(loss)
            total_valid = torch.sum(combined_mask.float())
            return total_loss / torch.clamp(total_valid, min=1e-7)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


class MaskedSparseCategoricalCrossEntropy(nn.Module):
    """
    PyTorch implementation of masked sparse categorical cross-entropy loss.

    This loss function ignores padding tokens (typically labeled as -100) and
    supports class weights for handling imbalanced datasets.

    Note: Consider using MaskedFocalLoss instead for better handling of
    class imbalance in PII detection tasks.
    """

    def __init__(
        self,
        pad_label: int = -100,
        class_weights: dict[int, float] | None = None,
        num_classes: int | None = None,
        reduction: str = "mean",
    ):
        """
        Initialize the masked loss function.

        Args:
            pad_label: Label value for padding tokens (HuggingFace standard: -100)
            class_weights: Dictionary mapping class IDs to weights
            num_classes: Total number of classes
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pad_label = pad_label
        self.class_weights = class_weights or {}
        self.num_classes = num_classes
        self.reduction = reduction

        if self.num_classes is not None:
            self._build_weight_tensor()

    def _build_weight_tensor(self):
        """Build a weight tensor from class weights dictionary."""
        weight_tensor = torch.ones(self.num_classes, dtype=torch.float32)
        for class_id, weight in self.class_weights.items():
            if 0 <= class_id < self.num_classes:
                weight_tensor[class_id] = float(weight)
        self.register_buffer("weight_tensor", weight_tensor)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the masked loss.

        Args:
            y_pred: Model predictions (logits) of shape (batch_size, seq_len, num_classes)
            y_true: True labels of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len) - combined with label mask

        Returns:
            Computed loss value
        """
        # Create mask for non-padded elements
        label_mask = y_true != self.pad_label

        # Combine with attention mask if provided
        if attention_mask is not None:
            mask = label_mask & (attention_mask.bool())
        else:
            mask = label_mask

        # Create safe version of y_true to avoid errors with negative labels
        y_true_safe = torch.where(mask, y_true, torch.zeros_like(y_true))

        # Compute cross-entropy loss
        loss = functional.cross_entropy(
            y_pred.view(-1, y_pred.size(-1)), y_true_safe.view(-1), reduction="none"
        )

        # Reshape loss to match input shape
        loss = loss.view(y_true.shape)

        # Apply class weights if available
        if hasattr(self, "weight_tensor"):
            weight_tensor = self.weight_tensor.to(y_true_safe.device)
            sample_weights = weight_tensor[y_true_safe]
            loss = loss * sample_weights

        # Apply padding mask
        loss = torch.where(mask, loss, torch.zeros_like(loss))

        # Apply reduction
        if self.reduction == "mean":
            total_loss = torch.sum(loss)
            total_valid = torch.sum(mask.float())
            return total_loss / torch.clamp(total_valid, min=1e-7)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning (PII detection + co-reference detection).
    """

    def __init__(
        self,
        pii_loss_fn: nn.Module,
        coref_loss_fn: nn.Module,
        pii_weight: float = 1.0,
        coref_weight: float = 1.0,
    ):
        """
        Initialize multi-task loss.

        Args:
            pii_loss_fn: Loss function for PII detection task
            coref_loss_fn: Loss function for co-reference detection task
            pii_weight: Weight for PII detection loss
            coref_weight: Weight for co-reference detection loss
        """
        super().__init__()
        self.pii_loss_fn = pii_loss_fn
        self.coref_loss_fn = coref_loss_fn
        self.pii_weight = pii_weight
        self.coref_weight = coref_weight

    def forward(
        self,
        pii_logits: torch.Tensor,
        pii_labels: torch.Tensor,
        coref_logits: torch.Tensor,
        coref_labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute combined multi-task loss.

        Args:
            pii_logits: PII detection logits (batch_size, seq_len, num_pii_classes)
            pii_labels: PII detection labels (batch_size, seq_len)
            attention_mask: Optional attention mask to exclude padded tokens (batch_size, seq_len)
            coref_logits: Co-reference detection logits (batch_size, seq_len, num_coref_classes)
            coref_labels: Co-reference detection labels (batch_size, seq_len)

        Returns:
            Combined loss value
        """
        pii_loss = self.pii_loss_fn(pii_logits, pii_labels, attention_mask)
        coref_loss = self.coref_loss_fn(coref_logits, coref_labels, attention_mask)

        total_loss = self.pii_weight * pii_loss + self.coref_weight * coref_loss
        return total_loss


class MultiTaskPIIDetectionModel(nn.Module):
    """
    Multi-task model for PII detection and co-reference detection.
    Uses a shared BERT encoder with two separate classification heads.
    """

    def __init__(
        self,
        model_name: str,
        num_pii_labels: int,
        num_coref_labels: int,
        id2label_pii: dict[int, str],
        id2label_coref: dict[int, str],
        classifier_dropout: float = 0.1,
    ):
        """
        Initialize multi-task model.

        Args:
            model_name: Name of the base BERT model
            num_pii_labels: Number of PII detection labels
            num_coref_labels: Number of co-reference detection labels
            id2label_pii: Mapping from PII label IDs to label names
            id2label_coref: Mapping from co-reference label IDs to label names
            classifier_dropout: Dropout rate for classification heads (default: 0.1)
        """
        super().__init__()

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Dropout layer for regularization
        self.dropout = nn.Dropout(classifier_dropout)

        # PII detection head
        self.pii_classifier = nn.Linear(hidden_size, num_pii_labels)

        # Co-reference detection head
        self.coref_classifier = nn.Linear(hidden_size, num_coref_labels)

        # Store label mappings
        self.num_pii_labels = num_pii_labels
        self.num_coref_labels = num_coref_labels
        self.id2label_pii = id2label_pii
        self.id2label_coref = id2label_coref

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pii_labels: torch.Tensor | None = None,
        coref_labels: torch.Tensor | None = None,
    ):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            pii_labels: PII labels for training (batch_size, seq_len)
            coref_labels: Co-reference labels for training (batch_size, seq_len)

        Returns:
            Dictionary with logits for both tasks
        """
        # Get shared encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # Apply dropout for regularization (only during training)
        sequence_output = self.dropout(sequence_output)

        # PII detection logits
        pii_logits = self.pii_classifier(sequence_output)

        # Co-reference detection logits
        coref_logits = self.coref_classifier(sequence_output)

        return {
            "pii_logits": pii_logits,
            "coref_logits": coref_logits,
            "hidden_states": sequence_output,
        }
