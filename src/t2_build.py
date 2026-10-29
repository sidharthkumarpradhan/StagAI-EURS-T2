"""Graph assembly steps that turn the shipped components into the submitted Track 2 ONNX.

swap_seg:     keep the restored_image subgraph of BOARD, take seg_logits from STUDENT (same input).
adapt_sharp:  append the per-image adaptive unsharp mask after restored_image.
Same code as tools/t2_swap_seg.py and tools/t2_adapt_sharp.py (which also print parity and checker output),
without onnxruntime, so it runs in the organizers' base image.
"""
import inspect
import os

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import compose, helper, utils


def _sub(path, output, tmp):
    m = onnx.load(path)
    utils.extract_model(path, tmp, [m.graph.input[0].name], [output])
    return onnx.load(tmp)


def swap_seg(board_path, student_path, out):
    board, student = onnx.load(board_path), onnx.load(student_path)
    assert board.graph.input[0].type == student.graph.input[0].type, "input shapes differ"
    rest = _sub(board_path, "restored_image", out + ".rest.tmp")
    seg = compose.add_prefix(_sub(student_path, "seg_logits", out + ".seg.tmp"), "s_", rename_inputs=False, rename_outputs=False)
    seg_in, rest_in = seg.graph.input[0].name, rest.graph.input[0].name
    for n in seg.graph.node:  # feed the student's nodes from the board's input; keep the official output name
        n.input[:] = [rest_in if i == seg_in else i for i in n.input]
        n.output[:] = ["seg_logits" if o == "s_seg_logits" else o for o in n.output]
    for o in seg.graph.output:
        o.name = "seg_logits" if o.name == "s_seg_logits" else o.name
    g = rest.graph
    g.node.extend(seg.graph.node)
    g.initializer.extend(seg.graph.initializer)
    g.value_info.extend(seg.graph.value_info)
    g.output.extend(seg.graph.output)
    opsets = {o.domain: o.version for o in board.opset_import}
    merged = helper.make_model(g, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()],
                               ir_version=board.ir_version)
    onnx.checker.check_model(merged, full_check=True)
    onnx.save(merged, out)
    for t in (out + ".rest.tmp", out + ".seg.tmp"):
        os.remove(t)


def _blur(r):
    """Gaussian sigma 1, 9 taps, BORDER_REFLECT_101 by static slices, as two depthwise convs."""
    k = torch.exp(-0.5 * (torch.arange(9, dtype=torch.float32) - 4) ** 2); k = k / k.sum()
    p = torch.cat([r[..., :, 1:5].flip(-1), r, r[..., :, -5:-1].flip(-1)], -1)
    p = torch.cat([p[..., 1:5, :].flip(-2), p, p[..., -5:-1, :].flip(-2)], -2)
    return F.conv2d(F.conv2d(p, k.view(1, 1, 1, 9).repeat(3, 1, 1, 1), groups=3), k.view(1, 1, 9, 1).repeat(3, 1, 1, 1), groups=3)


class AdaptSharp(nn.Module):
    """a = clip(ALPHA * (S0 - ln s), 0, AMAX) with s = mean |r - blur(r)|; output clip(r + a (r - blur(r)), 0, 1)."""

    def __init__(self, s0, alpha, amax):
        super().__init__()
        self.s0, self.alpha, self.amax = s0, alpha, amax

    def forward(self, r):
        d = r - _blur(r)
        s = d.abs().mean()
        a = (self.alpha * (self.s0 - torch.log(s + 1e-8))).clamp(0, self.amax)
        return (r + a * d).clamp(0, 1)


def adapt_sharp(base_path, out, s0=-4.7435, alpha=2.0, amax=4.0):
    base = onnx.load(base_path)
    h, w = [d.dim_value for d in base.graph.input[0].type.tensor_type.shape.dim][2:]
    opset = max(o.version for o in base.opset_import if o.domain in ("", "ai.onnx"))
    spath = out.replace(".onnx", "_sharp.onnx")
    # TorchScript exporter, as for the submitted file (torch >= 2.9 defaults to dynamo; torch 2.4 has no such argument)
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(AdaptSharp(s0, alpha, amax).eval(), (torch.rand(1, 3, h, w),), spath, input_names=["r_in"],
                      output_names=["r_out"], opset_version=opset, do_constant_folding=True, **kw)
    sh = onnx.load(spath)
    os.remove(spath)
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
    onnx.save(m, out, save_as_external_data=False)
