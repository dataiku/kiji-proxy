"""Model architecture and loss functions."""

import torch
from torch import nn
from torch.nn import functional
from transformers import AutoModel


class MaskedSparseCategoricalCrossEntropy(nn.Module):
    """
    PyTorch implementation of masked sparse categorical cross-entropy loss.

    This loss function ignores padding tokens (typically labeled as -100) and
    supports class weights for handling imbalanced datasets.
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


class PIIDetectionModel(nn.Module):
    """
    Model for PII detection using a BERT encoder with a classification head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: dict[int, str],
        classifier_dropout: float = 0.1,
    ):
        """
        Initialize PII detection model.

        Args:
            model_name: Name of the base BERT model
            num_labels: Number of PII detection labels
            id2label: Mapping from label IDs to label names
            classifier_dropout: Dropout rate for classification head (default: 0.1)
        """
        super().__init__()

        # Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Dropout layer for regularization
        self.dropout = nn.Dropout(classifier_dropout)

        # PII detection head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Store label mappings
        self.num_labels = num_labels
        self.id2label = id2label

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Labels for training (batch_size, seq_len)

        Returns:
            Dictionary with logits
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # Apply dropout for regularization (only during training)
        sequence_output = self.dropout(sequence_output)

        # PII detection logits
        logits = self.classifier(sequence_output)

        return {
            "logits": logits,
            "hidden_states": sequence_output,
        }
