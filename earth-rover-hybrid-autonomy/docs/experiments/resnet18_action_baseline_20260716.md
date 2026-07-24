# ResNet18 Action Baseline

This record freezes the completed Phase 1-3 pipeline baseline at commit `7c8765a`. Generated data and binaries remain outside Git.

## Configuration

- Model: ImageNet-pretrained ResNet18 with a five-class head.
- Input: one front RGB frame, directly resized from 16:9 to `224x224` by the preserved baseline loader.
- Output order: `STOP`, `FORWARD`, `LEFT`, `RIGHT`, `REVERSE`.
- Labels: existing `linear`/`angular` threshold mapping from `training/datasets/action_labels.py`.
- Split: 10 train rides, 2 validation rides, and 2 held-out test rides; no ride overlap.
- Selected samples: train 2,500, validation 500, test 500.
- Training: weighted cross entropy, AdamW, learning rate `3e-4`, batch size 32, deterministic seed `20260716`, early stopping with patience 3.

Train class counts were STOP 687, FORWARD 1,242, LEFT 223, RIGHT 215, and REVERSE 133. The best checkpoint was selected at epoch 3 by validation macro F1.

## Results

- Validation macro F1: `0.4084`.
- Test accuracy: `0.3420`.
- Test balanced accuracy: `0.2762`.
- Test macro F1: `0.2694`.
- Test confusion matrix, rows and columns in output order:

```text
[[26, 40, 17, 79, 1],
 [3, 116, 18, 20, 12],
 [5, 43, 15, 8, 1],
 [2, 40, 6, 6, 7],
 [7, 12, 6, 2, 8]]
```

The external checkpoint is `resnet18_small_baseline_best.pt`; metrics are in `small_baseline_report.json`; held-out visualization is `held_out_test_predictions.mp4`. Their Dell experiment directory is `$HOME/datasets/outputs/frodobots_2k_phase3/small_baseline/` and is not committed.

## Conclusion

The model is retained as a reusable decoding, training, checkpoint, metric, and visualization baseline. It is not a production controller. Route intent is not observable from one image, and abrupt human route choices can make a visually reasonable prediction disagree with the recorded command. The new traversability workflow is separate and does not alter this baseline.
