# StagAI: EURS @ MaCVi WACV 2027, Track 2 reproduction

Reported testing-phase submission: **19409** (`d9cc10e43ef5`, composite 0.419), Jetson AGX Orin, ONNX.

```
repro.yaml                 team StagAI, track eurs-t2, submission 19409
METHOD.md                  approach, training data, external data
export_onnx.py             main(model_dir, out_onnx): regenerates the submitted ONNX from weights/
src/t2_build.py            the two graph-assembly steps export_onnx.py runs
weights/                   StagAI_track2_base.onnx, t2DAfxall.onnx (+ the .pt checkpoints and configs they come from)
onnx/submitted_model.onnx  the exact file uploaded for 19409 (sha256 8be4c751...2080)
tools/                     the original build scripts (graft, swap, sharpening), for provenance
repro_check.py, docker/    from the organizers' template
```

```bash
python repro_check.py . --export-onnx      # regenerates the ONNX and compares it with onnx/submitted_model.onnx
```

Code: MIT (see LICENSE). Weights: trained by team StagAI; see METHOD.md for the pretrained models and data used.
