"""Smoke tests for EnhancerSegmenter (random encoder, no pretrained weights)."""

import torch

from bolinas.enhancer_classification.model import ENCODER_OUTPUT_DIM
from bolinas.enhancer_segmentation.model import EnhancerSegmenter

BIN_SIZE = 128
NUM_BINS = 128
SEQ_LEN = BIN_SIZE * NUM_BINS  # 16384
BATCH = 2


def _random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, 4, (BATCH, SEQ_LEN))
    x = torch.nn.functional.one_hot(indices, num_classes=4).float()
    y = torch.zeros(BATCH, NUM_BINS)
    y[0, 10:12] = 1.0
    return x, y


def test_forward_output_shape():
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False)
    model.eval()
    x, _ = _random_batch()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, NUM_BINS)


def test_loss_computation():
    """Masked BCE returns a scalar loss; per-bin shape comes from
    reduction='none' which is then masked + averaged in _masked_bce."""
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False, pos_weight=10.0)
    x, y = _random_batch()
    logits = model(x)
    # Per-bin BCE has shape (B, num_bins) when reduction='none'.
    per_bin = model.loss_fn(logits, y)
    assert per_bin.shape == (BATCH, NUM_BINS)
    # _masked_bce reduces to a scalar over the labeled bins.
    loss = model._masked_bce(logits, y)
    assert loss.shape == ()
    assert loss.item() > 0


def test_masked_bce_matches_mean_on_binary_labels():
    """On binary {0, 1} labels (no -1s), _masked_bce should match the
    original reduction='mean' BCE bit-for-bit."""
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False, pos_weight=10.0)
    x, y = _random_batch()
    logits = model(x)
    masked = model._masked_bce(logits, y)
    mean_bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(10.0), reduction="mean"
    )
    expected = mean_bce(logits, y)
    assert torch.allclose(masked, expected, atol=1e-7)


def test_masked_bce_excludes_minus_one_bins():
    """Bins with label=-1 must drop out of the loss; the result must equal
    the loss computed only over the labeled subset."""
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False, pos_weight=10.0)
    x, y = _random_batch()
    # Mark a span of bins as gray-zone (-1) on both samples.
    y[:, 20:40] = -1.0
    logits = model(x)
    masked = model._masked_bce(logits, y)
    # Compute the same thing manually over only labeled bins.
    flat_mask = (y >= 0).bool()
    mean_bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(10.0), reduction="mean"
    )
    expected = mean_bce(logits[flat_mask], y.clamp(min=0)[flat_mask])
    assert torch.allclose(masked, expected, atol=1e-7)


def test_freeze_backbone():
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=True)
    for param in model.encoder.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_head_includes_groupnorm():
    """The head applies channel-wise normalization before the final projection
    (issue #115 follow-up: stabilize grad norm)."""
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False)
    # Head is Sequential[GroupNorm, Conv1d].
    assert isinstance(model.head[0], torch.nn.GroupNorm)
    assert model.head[0].num_channels == 1536
    assert isinstance(model.head[-1], torch.nn.Conv1d)


def test_per_species_metric_per_genome():
    """Per-species AUPRC metrics are created one per genome."""
    model = EnhancerSegmenter(
        weights_path=None,
        freeze_backbone=False,
        genomes=["homo_sapiens", "mus_musculus"],
    )
    assert set(model.val_auprc_per_species.keys()) == {"homo_sapiens", "mus_musculus"}


def test_transformer_path_forward_and_shape():
    """n_transformer_layers>0 inserts a truncated tower and produces the
    same output shape as the encoder-only path."""
    model = EnhancerSegmenter(
        weights_path=None, freeze_backbone=False, n_transformer_layers=1
    )
    model.eval()
    x, _ = _random_batch()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, NUM_BINS)
    # Tower exists and has exactly one block.
    assert model.tower is not None
    assert len(model.tower.tower.blocks) == 1
    # Species embedding is a single learnable 1536-vector (not nn.Embedding).
    assert model.tower.species_embed.shape == (ENCODER_OUTPUT_DIM,)
    assert model.tower.species_embed.requires_grad


def test_transformer_path_freeze_backbone():
    """freeze_backbone freezes encoder and tower (including species_embed);
    only the head remains trainable."""
    model = EnhancerSegmenter(
        weights_path=None, freeze_backbone=True, n_transformer_layers=1
    )
    for p in model.encoder.parameters():
        assert not p.requires_grad
    for p in model.tower.parameters():
        assert not p.requires_grad
    for p in model.head.parameters():
        assert p.requires_grad
