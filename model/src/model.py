"""Model architecture and loss functions."""

import contextlib

import torch
from torch import nn
from torch.nn import functional
from torchcrf import CRF
from transformers import AutoModel, PretrainedConfig


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


class PIIDetectionModel(nn.Module):
    """Model for PII detection using a BERT encoder with a classification head."""

    def __init__(
        self,
        model_name: str,
        num_pii_labels: int,
        id2label_pii: dict[int, str],
        *,
        encoder_config: PretrainedConfig | None = None,
    ):
        """
        Initialize PII detection model.

        Args:
            model_name: Name of the base BERT model
            num_pii_labels: Number of PII detection labels
            id2label_pii: Mapping from PII label IDs to label names
            encoder_config: Saved architecture for checkpoint loading; when supplied,
                initialize the encoder without downloading pretrained weights.
        """
        super().__init__()

        # Shared encoder
        self.encoder = (
            AutoModel.from_config(encoder_config)
            if encoder_config is not None
            else AutoModel.from_pretrained(model_name)
        )
        hidden_size = self.encoder.config.hidden_size

        # PII detection head (MLP with bottleneck)
        self.pii_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_pii_labels),
        )

        # CRF layer for valid BIO sequence decoding
        self.crf = CRF(num_pii_labels, batch_first=True)

        # Store label mappings
        self.num_pii_labels = num_pii_labels
        self.id2label_pii = id2label_pii

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pii_labels: torch.Tensor | None = None,
    ):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            pii_labels: PII labels for training (batch_size, seq_len)

        Returns:
            Dictionary with PII logits and hidden states
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # PII detection logits
        pii_logits = self.pii_classifier(sequence_output)

        result = {
            "pii_logits": pii_logits,
            "hidden_states": sequence_output,
        }

        # Compute CRF loss if PII labels are provided
        if pii_labels is not None:
            # Create mask: True for real tokens, False for padding (-100)
            crf_mask = pii_labels != -100
            # Replace -100 with 0 for CRF (must happen before mask override)
            safe_labels = torch.where(crf_mask, pii_labels, 0)
            # CRF requires the first timestep mask to be True for all sequences
            crf_mask[:, 0] = True
            # Same value as -self.crf(..., reduction="mean"), in far fewer ops
            crf_loss = -crf_log_likelihood(
                self.crf, pii_logits, safe_labels, crf_mask
            ).mean()
            result["crf_loss"] = crf_loss

        return result

    def decode(
        self, pii_logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[list[int]]:
        """Decode PII logits using Viterbi algorithm via CRF.

        Args:
            pii_logits: Emission scores (batch_size, seq_len, num_labels)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            List of predicted label sequences
        """
        return viterbi_decode_on_cpu(self.crf, pii_logits, attention_mask)


def viterbi_decode_on_cpu(
    crf: CRF, emissions: torch.Tensor, attention_mask: torch.Tensor
) -> list[list[int]]:
    """Viterbi-decode on CPU, whatever device the model is on.

    torchcrf backtracks with one ``.item()`` per token. On an accelerator each
    is a device-to-host sync (about 10,000 per eval batch), which made TPU
    evaluation run at about 20 rows/s. The emissions and the small transition
    matrices are copied to CPU once instead. The result is the same: the
    decode adds and compares in float32 on any device.
    """
    if crf.transitions.device.type != "cpu":
        cpu_crf = CRF(crf.num_tags, batch_first=crf.batch_first)
        cpu_crf.load_state_dict(
            {name: value.detach().cpu() for name, value in crf.state_dict().items()}
        )
        crf = cpu_crf
    return crf.decode(
        emissions.detach().float().cpu(), mask=attention_mask.detach().bool().cpu()
    )


def crf_log_likelihood(
    crf: CRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sequence log-likelihood of ``tags``, as torchcrf computes it.

    Returns the same values as ``crf(emissions, tags, mask, reduction="none")``
    for a batch-first CRF. torchcrf loops over positions in Python, about 54
    dispatched ops per token for forward and backward (13,839 at 256 tokens).
    On TorchTPU every eager op is its own XLA executable, so that loop made
    the training step host-bound. Here the gold-path score is one gather over
    all positions, and the partition function multiplies the per-position
    transfer matrices pairwise in a log2(length)-deep tree: 438 ops at 256
    tokens, 473 at 512. The price is batch * length * num_tags**3 multiply-adds
    (about 2.4 GFLOP for 32 x 256 x 53 tags) and a (batch, length, num_tags,
    num_tags) float32 buffer, both small for an accelerator.

    The matmuls must run in full float32. Under bf16 autocast they would move
    the result by about 1 nat, so autocast is disabled. TorchTPU sets the
    float32 matmul precision to "medium" (bf16 passes), which moved the loss by
    0.03 nats and the emission gradient by 0.014 on the v5 CRF, so the matmuls
    run at "highest" in both passes (``_Float32Matmul``).

    The first mask column must be all True, as torchcrf requires. It is not
    checked, because checking it would read a device tensor on the host.
    ``mask`` may have gaps: masked positions are skipped exactly as torchcrf
    skips them, including its quirks (see ``_gold_path_score``).

    Args:
        crf: A batch-first ``torchcrf.CRF``; its parameters are used as is.
        emissions: Emission scores (batch_size, seq_len, num_tags), any float dtype.
        tags: Gold tags (batch_size, seq_len), valid tag ids at every position.
        mask: Positions to score (batch_size, seq_len), bool or 0/1.

    Returns:
        Log-likelihood per sequence (batch_size,), in float32.
    """
    if not crf.batch_first:
        raise ValueError("crf_log_likelihood expects a batch-first CRF")
    # torchcrf adds float32 parameters to the emissions, which promotes bf16
    # emissions to float32. Upcast once so this matches it.
    emissions = emissions.float()
    mask = mask.bool()
    with _autocast_disabled(emissions.device.type):
        return _gold_path_score(crf, emissions, tags, mask) - _log_partition(
            crf, emissions, mask
        )


def _gold_path_score(
    crf: CRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Score of the gold path, torchcrf's ``_compute_score`` without the loop.

    torchcrf adds the transition into each unmasked position from the tag at
    the previous position, masked or not, and takes the last tag at index
    ``mask.sum() - 1``. Both differ from skipping masked positions when the
    mask has gaps; they are kept so the loss is unchanged.
    """
    weights = mask.to(emissions.dtype)
    emitted = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
    moved = crf.transitions[tags[:, :-1], tags[:, 1:]]
    last_tags = tags.gather(1, mask.sum(1, keepdim=True) - 1).squeeze(1)
    return (
        crf.start_transitions[tags[:, 0]]
        + (emitted * weights).sum(1)
        + (moved * weights[:, 1:]).sum(1)
        + crf.end_transitions[last_tags]
    )


def _log_partition(
    crf: CRF, emissions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Log partition function, torchcrf's ``_compute_normalizer`` without the loop.

    The forward algorithm is ``alpha_t = alpha_{t-1} @ M_t`` with
    ``M_t[i, j] = exp(transitions[i, j] + emissions[t, j])``, and ``M_t`` is
    the identity at masked positions. The product of all ``M_t`` is taken
    pairwise, halving the number of matrices at each level. Everything is
    exponentiated after subtracting a maximum, and each product is divided by
    its largest entry, so nothing overflows; the subtracted logs are added
    back to the result. Detaching those shifts leaves the value and the
    gradient unchanged and keeps them out of the backward graph.
    """
    batch_size, seq_len, num_tags = emissions.shape

    # Position 0 in log space, shifted so the largest entry is 0.
    alpha = crf.start_transitions + emissions[:, 0]
    log_z = alpha.detach().amax(1)
    alpha = (alpha - log_z.unsqueeze(1)).exp().unsqueeze(1)

    # Transfer matrix for each later position: (batch_size, seq_len - 1, K, K).
    keep = mask[:, 1:]
    emit = emissions[:, 1:]
    emit_shift = emit.detach().amax(2)
    trans_shift = crf.transitions.detach().amax()
    steps = (crf.transitions - trans_shift).exp() * (
        emit - emit_shift.unsqueeze(2)
    ).exp().unsqueeze(2)
    eye = torch.eye(num_tags, dtype=steps.dtype, device=steps.device)
    steps = torch.where(keep[:, :, None, None], steps, eye)
    log_z = log_z + ((emit_shift + trans_shift) * keep).sum(1)

    # Pad with identities to a power of two, then multiply neighbours pairwise.
    count = seq_len - 1
    padded = 1 << max(count - 1, 0).bit_length()
    if padded > count:
        pad = eye.expand(batch_size, padded - count, num_tags, num_tags)
        steps = torch.cat([steps, pad], 1)
    while steps.size(1) > 1:
        steps = steps.unflatten(1, (-1, 2))
        steps = _Float32Matmul.apply(steps[:, :, 0], steps[:, :, 1])
        scale = steps.detach().amax((2, 3), keepdim=True)
        steps = steps / scale
        log_z = log_z + scale.log().sum((1, 2, 3))

    end_shift = crf.end_transitions.detach().amax()
    end = (crf.end_transitions - end_shift).exp()
    total = (_Float32Matmul.apply(alpha, steps[:, 0]).squeeze(1) * end).sum(1)
    return log_z + end_shift + total.log()


class _Float32Matmul(torch.autograd.Function):
    """``a @ b`` for same-shaped batches, at "highest" float32 matmul precision.

    The precision is a global setting that autograd reads again when it runs
    the backward pass, after the forward has returned, so a context manager
    around the forward pass alone would leave the gradient matmuls at the
    backend's default. This sets it in both passes and restores it after.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        with _highest_matmul_precision():
            return a @ b

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        with _highest_matmul_precision():
            if ctx.needs_input_grad[0]:
                grad_a = grad @ b.transpose(-2, -1)
            if ctx.needs_input_grad[1]:
                grad_b = a.transpose(-2, -1) @ grad
        return grad_a, grad_b


@contextlib.contextmanager
def _highest_matmul_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _autocast_disabled(device_type: str):
    """Keep the CRF's matmuls in float32 under bf16 mixed precision."""
    if torch.amp.is_autocast_available(device_type):
        return torch.autocast(device_type, enabled=False)
    return contextlib.nullcontext()
