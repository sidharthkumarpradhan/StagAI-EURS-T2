# Track 2 graft (15 Oct): keep a board student's seg_logits subgraph exactly and replace its restored_image with an
# AFFINE=1 restorer from t1_restnet2 (low-res UNet -> per-pixel colour affine applied at full resolution).
# Why: rnK2 (pixels-only restorer recipe) reached 24.1 dB held-out vs the student's 20.7 dB, and the student's own
# restorer + refiner cost ~1.9 GFLOPs that the graft frees. The seg branch (board q 0.558) is untouched.
# Env: STUDENT (onnx), CKPT (restnet .pt), BASE, LOWRES, LEVELS (as the restnet run), OUT (onnx path).
# Inputs to the restorer are built in-graph exactly as t1_restnet2 with NOKEY=1: [x, x, 0 x 11, gstats x 12], where
# gstats = per-channel mean, std, 1st and 99th percentile. Percentiles use TopK on a 2x area-downsampled copy (TensorRT
# caps TopK k at 3840; full res would need k ~ 7.9k); STATS_CHECK reports the difference to numpy's full-res values.
# Prints PARITY (ORT grafted seg vs student seg, restored vs torch), the official checker output, then DONE.
import os, sys, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import onnx, onnxruntime as ort
from onnx import helper

sys.path.insert(0, "/kaggle/tmp/repo/src")
from macvi.eurs import onnx_tools

STUDENT, CKPT, OUT = os.environ["STUDENT"], os.environ["CKPT"], os.environ["OUT"]
os.environ.setdefault("AFFINE", "1"); os.environ.setdefault("NOKEY", "1")
SRC = os.path.expanduser("~/macvi_run/t1_restnet2.py")
g = {"__name__": "r2", "__file__": SRC}
sys.path.insert(0, os.path.dirname(SRC))
exec(compile(open(SRC).read().split("if __name__")[0], "r2", "exec"), g)
AMOUNT = float(os.environ.get("AMOUNT", "0"))  # unsharp amount chosen on the held-out VAL8 sweep of the restorer
H, W = [d.dim_value for d in onnx.load(STUDENT).graph.input[0].type.tensor_type.shape.dim][2:]


def quantile_topk(x, q):
    """q-quantile per channel of (1, 3, N) as numpy's linear interpolation, via TopK (k <= 3840)."""
    n = x.shape[-1]
    pos = q * (n - 1)
    lo = int(np.floor(pos)); frac = pos - lo
    if q < 0.5:
        v = -torch.topk(-x, lo + 2, dim=-1).values  # ascending smallest lo+2
        a, b = v[..., lo], v[..., lo + 1]
    else:
        k = n - lo  # largest values: index lo from the bottom is (n-1-lo) from the top
        v = torch.topk(x, k + 1, dim=-1).values
        a, b = v[..., k - 1], v[..., k - 2] if k >= 2 else v[..., k - 1]
    return a + (b - a) * frac


class Restorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = g["make_net"]()
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)
        self.net.load_state_dict(sd.get("net", sd) if isinstance(sd, dict) and "net" in sd else sd)
        # input channels 6-16 (water one-hot, depth) are always zero with NOKEY=1: drop them from the first conv so the
        # graph carries no (1, 11, H, W) zero constant (36 MB); zeros contribute exactly 0 to that conv
        c0 = self.net.enc[0][0]
        keep = list(range(6)) + list(range(17, c0.in_channels))
        new = nn.Conv2d(len(keep), c0.out_channels, c0.kernel_size, c0.stride, c0.padding)
        with torch.no_grad():
            new.weight.copy_(c0.weight[:, keep]); new.bias.copy_(c0.bias)
        self.net.enc[0][0] = new

    def stats(self, x):
        b = x.shape[0]
        flat = x.flatten(2)
        mean = flat.mean(-1)
        std = ((flat - mean[..., None]) ** 2).mean(-1).sqrt()
        small = F.avg_pool2d(x, 2).flatten(2)
        return torch.cat([mean, std, quantile_topk(small, 0.01), quantile_topk(small, 0.99)], 1).view(b, 12, 1, 1)

    def forward(self, x):
        b, h, w = int(x.shape[0]), int(x.shape[2]), int(x.shape[3])  # static shapes in the export
        st = self.stats(x).expand(b, 12, h, w)
        inp = torch.cat([x, x, st], 1)  # the 11 zero channels are folded out of the first conv (see __init__)
        return self.sharpen(self.net(inp))

    def sharpen(self, r):
        """AMOUNT > 0: unsharp mask as t1_rest_ab.unsharp (cv2 GaussianBlur sigma 1 = 9 taps, BORDER_REFLECT_101), in-graph
        as two depthwise convs on a reflect-padded copy (~0.09 GFLOPs at 768x1024)."""
        if AMOUNT == 0:
            return r
        k = torch.exp(-0.5 * (torch.arange(9, dtype=torch.float32) - 4) ** 2); k = k / k.sum()
        # BORDER_REFLECT_101 by static slices (an ONNX Pad gets data-dependent pads here, which leaves the conv shape unknown)
        p = torch.cat([r[..., :, 1:5].flip(-1), r, r[..., :, -5:-1].flip(-1)], -1)
        p = torch.cat([p[..., 1:5, :].flip(-2), p, p[..., -5:-1, :].flip(-2)], -2)
        bl = F.conv2d(F.conv2d(p, k.view(1, 1, 1, 9).repeat(3, 1, 1, 1), groups=3), k.view(1, 1, 9, 1).repeat(3, 1, 1, 1), groups=3)
        return (r + AMOUNT * (r - bl)).clamp(0, 1)


if __name__ == "__main__":
    rest = Restorer().eval()
    ref_net = g["make_net"]().eval()
    _sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    ref_net.load_state_dict(_sd.get("net", _sd) if isinstance(_sd, dict) and "net" in _sd else _sd)
    stu_opset = max(o.version for o in onnx.load(STUDENT).opset_import if o.domain in ("", "ai.onnx"))
    rpath = OUT.replace(".onnx", "_restorer.onnx")
    torch.onnx.export(rest, (torch.rand(1, 3, H, W),), rpath, input_names=["input"], output_names=["restored_image"],
                      opset_version=stu_opset, do_constant_folding=True, dynamo=False)
    stu, res = onnx.load(STUDENT), onnx.load(rpath)
    in_name = stu.graph.input[0].name
    # prefix every restorer tensor except its input, then wire the input to the student's
    ren = {}
    for n in res.graph.node:
        for i, t in enumerate(n.input):
            if t and t != "input":
                n.input[i] = ren.setdefault(t, "rg_" + t)
            elif t == "input":
                n.input[i] = in_name
        for i, t in enumerate(n.output):
            n.output[i] = ren.setdefault(t, "rg_" + t)
        n.name = "rg_" + n.name
    for n in res.graph.node:  # .float() exports as Cast(to=FLOAT) on fp32 tensors: no-ops that break the fp16 check
        if n.op_type == "Cast" and n.attribute[0].i == onnx.TensorProto.FLOAT:
            n.op_type = "Identity"; del n.attribute[:]
    for init in res.graph.initializer:
        init.name = ren.setdefault(init.name, "rg_" + init.name)
    out_r = ren["restored_image"]
    graph = stu.graph
    keep_out = [o for o in graph.output if o.name == "seg_logits"]
    # drop the student's restored_image producer chain: rename its output, then prune unreachable nodes
    nodes = list(graph.node) + list(res.graph.node)
    for n in nodes:
        for i, t in enumerate(n.output):
            if t == "restored_image":
                n.output[i] = "old_restored_image"
            elif t == out_r:
                n.output[i] = "restored_image"
    need, kept = {"seg_logits", "restored_image"}, []
    for n in reversed(nodes):
        if any(o in need for o in n.output):
            kept.append(n); need.update(t for t in n.input if t)
    kept.reverse()
    inits = [i for i in list(graph.initializer) + list(res.graph.initializer) if i.name in need]
    rest_out = helper.make_tensor_value_info("restored_image", onnx.TensorProto.FLOAT, [1, 3, H, W])
    new = helper.make_graph(kept, "t2_graft", list(graph.input), [rest_out] + keep_out, inits)
    m = helper.make_model(new, opset_imports=[helper.make_opsetid("", stu_opset)])
    m.ir_version = stu.ir_version
    m = onnx.shape_inference.infer_shapes(m, data_prop=True)
    onnx.checker.check_model(m)
    onnx.save(m, OUT, save_as_external_data=False)

    # parity on a few real images (numpy full-res stats as in training vs the in-graph TopK stats)
    x = np.random.RandomState(0).rand(1, 3, H, W).astype(np.float32)
    imgs = [x]
    vdir = os.environ.get("IMGS")
    if vdir:
        import glob, cv2
        ps = sorted(glob.glob(vdir))
        for p in ps[::max(1, len(ps) // 6)][:6]:
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs.append((im.astype(np.float32) / 255).transpose(2, 0, 1)[None])
    s_ort = ort.InferenceSession(STUDENT, providers=["CPUExecutionProvider"])
    g_ort = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    out = {}
    for j, im in enumerate(imgs):
        seg_s = s_ort.run(["seg_logits"], {in_name: im})[0]
        r_g, seg_g = g_ort.run(["restored_image", "seg_logits"], {in_name: im})
        with torch.no_grad():
            u8 = (im[0].transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
            st = torch.from_numpy(g["gstats_p"](u8).astype(np.float32)).view(1, 12, 1, 1).expand(1, 12, H, W)
            t = torch.from_numpy(im)
            ref = ref_net(torch.cat([t, t, torch.zeros(1, 11, H, W), st], 1)).numpy()  # original 29-channel net
            if AMOUNT:  # reference unsharp exactly as the restorer's held-out scoring (cv2, float32)
                from t1_rest_ab import unsharp
                ref = unsharp(ref[0].transpose(1, 2, 0), AMOUNT).transpose(2, 0, 1)[None]
            st_g = rest.stats(t).flatten().numpy()
        out[j] = {"seg_maxdiff": float(np.abs(seg_s - seg_g).max()), "rest_vs_trainstats_maxdiff": float(np.abs(r_g - ref).max()),
                  "rest_vs_trainstats_psnr": float(10 * np.log10(1 / max(np.mean((r_g - ref) ** 2), 1e-12))),
                  "stats_maxdiff": float(np.abs(st_g - st[0, :, 0, 0].numpy()).max())}
    print("PARITY", json.dumps(out), flush=True)
    print("CHECK", onnx_tools.check(OUT), flush=True)
    print("OFFICIAL", onnx_tools.run_official_checker(OUT, os.path.expanduser("~/macvi_run")), flush=True)
    print("FP16", json.dumps(onnx_tools.fp16_parity(OUT, imgs[1:] or imgs)), flush=True)
    print("DONE", OUT)
