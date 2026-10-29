# Method description

Team: StagAI   Track(s): both (this repository: Track 2)   Reported testing-phase submission: 19409 (`d9cc10e43ef5`)

Sidharth Kumar Pradhan, Sidike Paheding. Related workshop paper: *TiDE-URS: A Teacher-Distilled Efficient Network for
Underwater Restoration and Segmentation* (MaCVi 2027 submission #21).

| File | sha256 |
|---|---|
| `onnx/submitted_model.onnx` (uploaded for 19409) | `8be4c7514cb86808fda6739bda989414f4256fc94468476c882368bffd962080` |

## Approach

One ONNX graph, input `input` (1, 3, 768, 1024) RGB in [0, 1], outputs `restored_image` (1, 3, 768, 1024) and raw
`seg_logits` (1, 5, 768, 1024). Opset 18, static shapes, 0.7725 M parameters, 10.8157 GFLOPs (official checker: PASSED).
Two separately trained networks share the input:

**Segmentation student** (`weights/t2DAfxall_best.pt`, exported as `weights/t2DAfxall.onnx`).
- MobileNetV3-Small (width 0.75, ImageNet init) at output stride 8, an LR-ASPP head with 96 channels and a stride-4
  classifier; logits upsampled in-graph to full resolution.
- Base student `t2E9_all_lv_ref8` (`weights/t2E9_all_lv_ref8_best.pt`): distilled online from a teacher (frozen DINOv3
  ViT-L/16 + FPN head, trained on all 20 train scenes): KL at temperature 2, class-weighted cross-entropy and
  Lovász-softmax. 60 epochs x 4,000 samples, batch 8, AdamW lr 2e-3, wd 1e-4, EMA 0.999; random rescale [0.5, 2],
  crop, flip, re-degradation at a fresh water type (p = 0.3); half of each batch from external photos. A per-class logit
  bias (calibrated on train data) is folded into the classifier bias.
- Fine-tune `t2DAfxall` (the one in the submitted file): 8 epochs x 2,000 samples, lr 5e-4, last epoch kept, all 20
  scenes, initialised from `t2E9_all_lv_ref8`, same online teacher plus feature distillation (1 - cosine to the teacher's
  DINOv3 features, weight 1.0) and 4x sampling of external photos that contain elasmobranch (3) or other (4).

**Restorer** (`weights/rnKHall.pt`): a 3-level UNet (base width 16) on a 1/4-resolution area-downsampled copy predicts a
per-pixel 3x3 colour matrix + bias, upsampled bilinearly and applied to the full-resolution input. It also sees 12
global statistics of the image computed in-graph (per-channel mean, std, 1st and 99th percentile via TopK on a 2x
downsampled copy); a fixed unsharp mask (amount 0.6, Gaussian sigma 1) follows. 120k steps, AdamW lr 3e-4, 384 crops,
loss log10(MSE) + 2 (1 - SSIM), all 20 train scenes plus 8 crops per step re-degraded from train clean images.

**Graph assembly.**
1. `tools/t2_graft.py` exports the restorer and replaces the base student's restoration branch with it
   (segmentation kept bit-exact) -> `weights/StagAI_track2_base.onnx` (sha256 `11f86821...2411`; this was our
   upload `d677e0b0db90`). Rebuilt byte-identically from the checkpoints on 25 Oct 2026; it needs our training-repo
   helper `t1_restnet2.py` and is shipped as provenance, not called by `export_onnx.py`.
2. `export_onnx.py` (= `tools/t2_swap_seg.py` + `tools/t2_adapt_sharp.py`): the `seg_logits` subgraph is replaced by the
   fine-tuned student `weights/t2DAfxall.onnx`, then a per-image adaptive unsharp step is appended to `restored_image`:
   s = mean |r - blur(r)| (Gaussian sigma 1, 9 taps, reflect-101), a = clip(2 (-4.7435 - ln s), 0, 4),
   output clip(r + a (r - blur(r)), 0, 1). The constants were fitted on held-out folds of the train scenes.

## Training data

The 20 scenes of `eurs_train.zip` only (4,000 degraded renderings with clean image and mask). No development or
testing-phase image was used for training, fine-tuning, pseudo-labelling or calibration; no weights were changed using
them. Disclosure on selection: (a) development-phase leaderboard scores guided which recipes we kept; (b) before
uploading 19409 we compared five candidate Track 2 graphs (all already trained) by their segmentation agreement with our
own Track 1 predicted masks on 80 Track 1 testing-phase images (no ground truth; nothing was trained or tuned on them,
and each output still depends only on its own input). The reported submission was then chosen by its testing-phase
leaderboard score (0.419, best of our five Track 2 testing-phase submissions).

**External data and pretrained models.**
- Segmentation masks (mapped to the five EURS classes, re-degraded synthetically with an image-formation model fitted on
  the EURS train pairs; near-duplicates of EURS images removed by perceptual hash): USIS16K, UIIS10K, USIS10K, SUIM,
  Coralscapes. Pixels the external masks leave unannotated were pseudo-labelled by our train-only teachers
  (kept at probability >= 0.6, otherwise ignored).
- ImageNet-pretrained MobileNetV3-Small (torchvision / timm) for the student.
- DINOv3 ViT-L/16 (LVD-1689M, `timm/vit_large_patch16_dinov3.lvd1689m`, DINOv3 License) inside the training-only teacher.
  It is not part of the ONNX.

## Anything non-standard at inference

None outside the ONNX: the graph takes the image alone (no metadata), and every output depends on its own input only.
The adaptive sharpening is part of the graph.

## Reproduction notes

`python repro_check.py . --export-onnx` regenerates the ONNX in a few seconds on CPU. Checked 29 Oct 2026 with
torch 2.5.1 / onnx 1.23.1 (the versions that produced the upload) and a newer torch: byte-identical to
`onnx/submitted_model.onnx`. Choice made from testing-phase leaderboard scores: we report 19409 (`d9cc10e43ef5`,
0.419) as our best of five testing-phase Track 2 submissions (see "Training data" for the selection disclosure).
