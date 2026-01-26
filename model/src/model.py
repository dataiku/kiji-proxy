"""Model architecture and loss functions."""

import torch
from torch import nn
from torch.nn import functional
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked loss.

        Args:
            y_pred: Model predictions (logits) of shape (batch_size, seq_len, num_classes)
            y_true: True labels of shape (batch_size, seq_len)

        Returns:
            Computed loss value
        """
        # Create mask for non-padded elements
        mask = y_true != self.pad_label

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
    ) -> torch.Tensor:
        """
        Compute combined multi-task loss.

        Args:
            pii_logits: PII detection logits (batch_size, seq_len, num_pii_classes)
            pii_labels: PII detection labels (batch_size, seq_len)
            coref_logits: Co-reference detection logits (batch_size, seq_len, num_coref_classes)
            coref_labels: Co-reference detection labels (batch_size, seq_len)

        Returns:
            Combined loss value
        """
        pii_loss = self.pii_loss_fn(pii_logits, pii_labels)
        coref_loss = self.coref_loss_fn(coref_logits, coref_labels)

        total_loss = self.pii_weight * pii_loss + self.coref_weight * coref_loss
        return total_loss


class MultiTaskPIIConfig(PretrainedConfig):
    """Configuration class for MultiTaskPIIDetectionModel."""

    model_type = "multitask_pii"

    def __init__(
        self,
        base_model_name: str = "answerdotai/ModernBERT-base",
        num_pii_labels: int = 49,
        num_coref_labels: int = 5,
        id2label_pii: dict[int, str] | None = None,
        id2label_coref: dict[int, str] | None = None,
        hidden_size: int = 768,
        **kwargs,
    ):
        """
        Initialize config.

        Args:
            base_model_name: Name of the base encoder model
            num_pii_labels: Number of PII detection labels
            num_coref_labels: Number of co-reference detection labels
            id2label_pii: Mapping from PII label IDs to label names
            id2label_coref: Mapping from co-reference label IDs to label names
            hidden_size: Hidden size of the encoder
            **kwargs: Additional arguments for PretrainedConfig
        """
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_pii_labels = num_pii_labels
        self.num_coref_labels = num_coref_labels
        self.id2label_pii = id2label_pii or {}
        self.id2label_coref = id2label_coref or {}
        self.hidden_size = hidden_size


class MultiTaskPIIDetectionModel(PreTrainedModel):
    """
    Multi-task model for PII detection and co-reference detection.
    Uses a shared BERT encoder with two separate classification heads.

    Inherits from PreTrainedModel for proper HuggingFace integration,
    including save_pretrained/from_pretrained and ONNX export support.
    """

    config_class = MultiTaskPIIConfig

    def __init__(self, config: MultiTaskPIIConfig):
        """
        Initialize multi-task model from config.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.encoder.config.hidden_size

        # Update config with actual hidden size from encoder
        config.hidden_size = hidden_size

        # PII detection head
        self.pii_classifier = nn.Linear(hidden_size, config.num_pii_labels)

        # Co-reference detection head
        self.coref_classifier = nn.Linear(hidden_size, config.num_coref_labels)

        # Store label mappings for convenience
        self.num_pii_labels = config.num_pii_labels
        self.num_coref_labels = config.num_coref_labels
        self.id2label_pii = config.id2label_pii
        self.id2label_coref = config.id2label_coref

        # Initialize weights for classification heads
        self.post_init()

    @classmethod
    def from_pretrained_legacy(
        cls,
        model_name: str,
        num_pii_labels: int,
        num_coref_labels: int,
        id2label_pii: dict[int, str],
        id2label_coref: dict[int, str],
    ):
        """
        Create model with legacy API (for backward compatibility with trainer).

        Args:
            model_name: Name of the base BERT model
            num_pii_labels: Number of PII detection labels
            num_coref_labels: Number of co-reference detection labels
            id2label_pii: Mapping from PII label IDs to label names
            id2label_coref: Mapping from co-reference label IDs to label names

        Returns:
            Initialized model
        """
        config = MultiTaskPIIConfig(
            base_model_name=model_name,
            num_pii_labels=num_pii_labels,
            num_coref_labels=num_coref_labels,
            id2label_pii=id2label_pii,
            id2label_coref=id2label_coref,
        )
        return cls(config)

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

        # PII detection logits
        pii_logits = self.pii_classifier(sequence_output)

        # Co-reference detection logits
        coref_logits = self.coref_classifier(sequence_output)

        return {
            "pii_logits": pii_logits,
            "coref_logits": coref_logits,
            "hidden_states": sequence_output,
        }
