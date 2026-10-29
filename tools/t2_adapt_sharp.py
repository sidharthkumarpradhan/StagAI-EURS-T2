# Track 2 restoration (27 Oct): per-image extra unsharp appended to a built ONNX (t2G3) after its restored_image.
# Sweep 26 Oct (t2_sharp_sweep_v100.py, fold 0): one extra amount for all images is best at 0, but murky water gains
# (w9-10 +0.011 at amount 2) and clear water loses. The amount is set per image from a no-GT statistic of the model's own
# output s (STAT 0: mean |r - blur(r)|, STAT 2: std of the grey image): a = clip(ALPHA * (S0 - log s), 0, AMAX), fitted on
# held-out folds (fitsharp). Same blur as the graft's sharpen(): Gaussian sigma 1, 9 taps, reflect-101, two depthwise convs.
# Env: BASE (onnx), OUT, S0, ALPHA, AMAX, STAT. Prints PARITY vs the numpy reference, the official checker, then DONE.
import os, sys, json
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
import onnx, onnxruntime as ort
from onnx import helper

sys.path.insert(0, "/kaggle/tmp/repo/src")
from macvi.eurs import onnx_tools

BASE, OUT = os.environ["BASE"], os.environ["OUT"]
S0, ALPHA, AMAX = (float(os.environ[k]) for k in ("S0", "ALPHA", "AMAX"))
STAT = int(os.environ.get("STAT", "0"))
H, W = [d.dim_value for d in onnx.load(BASE).graph.input[0].type.tensor_type.shape.dim][2:]


def blur(r):
    k = torch.exp(-0.5 * (torch.arange(9, dtype=torch.float32) - 4) ** 2); k = k / k.sum()
    p = torch.cat([r[..., :, 1:5].flip(-1), r, r[..., :, -5:-1].flip(-1)], -1)
    p = torch.cat([p[..., 1:5, :].flip(-2), p, p[..., -5:-1, :].flip(-2)], -2)
    return F.conv2d(F.conv2d(p, k.view(1, 1, 1, 9).repeat(3, 1, 1, 1), groups=3), k.view(1, 1, 9, 1).repeat(3, 1, 1, 1), groups=3)


class AdaptSharp(nn.Module):
    def forward(self, r):
        d = r - blur(r)
        if STAT == 0:
            s = d.abs().mean()
        else:
            gray = r.mean(1)
            s = ((gray - gray.mean()) ** 2).mean().sqrt()
        a = (ALPHA * (S0 - torch.log(s + 1e-8))).clamp(0, AMAX)
        return (r + a * d).clamp(0, 1)


def reference(r):  # numpy, as t2_sharp_sweep_v100 (HWC float32)
    bl = cv2.GaussianBlur(r, (9, 9), 1.0, borderType=cv2.BORDER_REFLECT_101)
    s = np.abs(r - bl).mean() if STAT == 0 else r.mean(2).std()
    a = np.clip(ALPHA * (S0 - np.log(s + 1e-8)), 0, AMAX)
    return np.clip(r + a * (r - bl), 0, 1), a


if __name__ == "__main__":
    base = onnx.load(BASE)
    opset = max(o.version for o in base.opset_import if o.domain in ("", "ai.onnx"))
    spath = OUT.replace(".onnx", "_sharp.onnx")
    torch.onnx.export(AdaptSharp().eval(), (torch.rand(1, 3, H, W),), spath, input_names=["r_in"], output_names=["r_out"],
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    sh = onnx.load(spath)
    for n in base.graph.node:
        n.output[:] = ["as_r0" if t == "restored_image" else t for t in n.output]
    for n in sh.graph.node:
        n.input[:] = ["as_r0" if t == "r_in" else ("as_" + t if t else t) for t in n.input]
        n.output[:] = ["restored_image" if t == "r_out" else "as_" + t for t in n.output]
        n.name = "as_" + n.name
        if n.op_type == "Cast" and n.attribute and n.attribute[0].i == onnx.TensorProto.FLOAT:
            n.op_type = "Identity"; del n.attribute[:]
    for i in sh.graph.initializer:
        i.name = "as_" + i.name
    g = helper.make_graph(list(base.graph.node) + list(sh.graph.node), "t2_adapt", list(base.graph.input),
                          list(base.graph.output), list(base.graph.initializer) + list(sh.graph.initializer))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = base.ir_version
    m = onnx.shape_inference.infer_shapes(m, data_prop=True)
    onnx.checker.check_model(m)
    onnx.save(m, OUT, save_as_external_data=False)

    b_ort = ort.InferenceSession(BASE, providers=["CPUExecutionProvider"])
    o_ort = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    iname = base.graph.input[0].name
    imgs = [np.random.RandomState(0).rand(1, 3, H, W).astype(np.float32)]
    import glob
    ps = sorted(glob.glob(os.environ.get("IMGS", "")))
    for p in ps[::max(1, len(ps) // 8)][:8]:
        im = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)
        imgs.append((im.astype(np.float32) / 255).transpose(2, 0, 1)[None])
    out = []
    for im in imgs:
        r0, seg0 = b_ort.run(["restored_image", "seg_logits"], {iname: im})
        r1, seg1 = o_ort.run(["restored_image", "seg_logits"], {iname: im})
        ref, a = reference(r0[0].transpose(1, 2, 0))
        out.append({"a": round(float(a), 3), "rest_maxdiff": float(np.abs(r1[0].transpose(1, 2, 0) - ref).max()),
                    "seg_maxdiff": float(np.abs(seg0 - seg1).max())})
    print("PARITY", json.dumps(out), flush=True)
    print("CHECK", onnx_tools.check(OUT), flush=True)
    print("OFFICIAL", onnx_tools.run_official_checker(OUT, os.path.expanduser("~/macvi_run")), flush=True)
    print("FP16", json.dumps(onnx_tools.fp16_parity(OUT, imgs[1:] or imgs)), flush=True)
    print("DONE", OUT)
