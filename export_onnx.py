"""Track 2: regenerate the submitted ONNX (testing-phase submission 19409, d9cc10e43ef5) from weights/.

The organizers call   main(model_dir, out_onnx).

  1. weights/StagAI_track2_base.onnx  (restorer rnKHall grafted onto student t2E9_all_lv_ref8, built by tools/t2_graft.py)
     gets its seg_logits subgraph replaced by the fine-tuned student weights/t2DAfxall.onnx      (src.t2_build.swap_seg)
  2. the per-image adaptive unsharp mask is appended after restored_image                     (src.t2_build.adapt_sharp)

Static input (1, 3, 768, 1024) in [0, 1]; outputs restored_image and seg_logits; 10.8157 GFLOPs, 0.7725 M parameters.
With torch 2.5.1 / onnx 1.23.1 the result is byte-identical to onnx/submitted_model.onnx (sha256 8be4c751...2080).
"""
import os
import tempfile

import torch

from src.t2_build import adapt_sharp, swap_seg


def main(model_dir, out_onnx):
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(os.path.abspath(out_onnx)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        swapped = os.path.join(tmp, "swap.onnx")
        swap_seg(os.path.join(model_dir, "StagAI_track2_base.onnx"), os.path.join(model_dir, "t2DAfxall.onnx"), swapped)
        adapt_sharp(swapped, out_onnx, s0=-4.7435, alpha=2.0, amax=4.0)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "weights", sys.argv[2] if len(sys.argv) > 2 else "out/export.onnx")
