# Hybrid CNN → SNN backbone (design TODO)

This repository currently supports `backbone_type` values `spiking_resnet1d` and `cnn_resnet1d` as separate stacks. A **hybrid** path (continuous stem → spiking temporal trunk) was deferred to keep checkpoints, configs, and training scripts stable.

## Target architecture

- **Input** `[B, C, T]` after optional feature stacking.
- **CNN stem**: small `Conv1d + BN + ReLU` stack (or reuse `CNNStem1d`) producing `[B, C', T']` with preserved temporal resolution (stride 1), or mild downsampling if matched to existing SNN stem expectations.
- **SNN trunk**: existing `SpikingBasicBlock1d` stages + LIF, taking continuous activations only if we insert a **rate-coded** or **membrane-input** interface (today’s `SpikingStem1d` expects raw sensor channels; hybrid would need a `HybridStem` that maps `C'` continuous channels to first spiking stage).
- **Pooling / head**: reuse `mean` / `adaptive` / `attention` and `LinearProbeHead` / `MLPProbeHead`.

## Config sketch

```yaml
backbone_type: hybrid_cnn_snn
hybrid:
  cnn_stem_channels: 32
  snn_base_channels: 32
```

## Implementation steps (future)

1. Add `models/hybrid_cnn_snn1d.py` composing CNN stem + adapted spiking blocks; expose `out_dim` like existing backbones.
2. Extend `train/snn_common.py` `build_snn_backbone` (or parallel `build_hybrid_backbone`) and `snn_model_cfg_from_yaml` with `backbone_type` dispatch.
3. SSL / probe loaders: ensure `model_cfg["backbone_type"]` round-trips in checkpoints; add migration note for old SSL weights (not loadable into hybrid without shape-matched stem).
4. Smoke test: forward `[B,3,T]` → embedding dim matches `out_dim`.

Until then, use **CNN-only** or **SNN-only** baselines and the new pooling / stacking / MLP-head options for performance gains.
