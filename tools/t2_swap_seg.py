# Track 2 (25 Oct 2026): swap the segmenter of a board graph without rebuilding its restorer.
# BOARD (e.g. graft/t2G3.onnx) keeps its restored_image subgraph exactly; STUDENT (a t2_distill ONNX with the same input
# shape) supplies seg_logits. Both subgraphs are extracted, the student's names get a prefix, and they share one input.
# Prints PARITY (max |diff| of each output vs its source graph on random inputs), then DONE. Env: BOARD, STUDENT, OUT.
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import compose, utils

BOARD, STUDENT, OUT = os.environ["BOARD"], os.environ["STUDENT"], os.environ["OUT"]


def sub(path, output, tmp):
    m = onnx.load(path)
    utils.extract_model(path, tmp, [m.graph.input[0].name], [output])
    return onnx.load(tmp)


board, student = onnx.load(BOARD), onnx.load(STUDENT)
assert board.graph.input[0].type == student.graph.input[0].type, "input shapes differ"
rest = sub(BOARD, "restored_image", OUT + ".rest.tmp")
seg = compose.add_prefix(sub(STUDENT, "seg_logits", OUT + ".seg.tmp"), "s_", rename_inputs=False, rename_outputs=False)
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
merged = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid(d, v) for d, v in opsets.items()],
                                ir_version=board.ir_version)
onnx.checker.check_model(merged, full_check=True)
onnx.save(merged, OUT)
for t in (OUT + ".rest.tmp", OUT + ".seg.tmp"):
    os.remove(t)

shape = [d.dim_value for d in board.graph.input[0].type.tensor_type.shape.dim]
x = np.random.default_rng(0).random(shape, dtype=np.float32)


def run(path):
    sess = ort.InferenceSession(path)
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {sess.get_inputs()[0].name: x})))



got, ref_b, ref_s = run(OUT), run(BOARD), run(STUDENT)
print("PARITY restored", float(np.abs(got["restored_image"] - ref_b["restored_image"]).max()),
      "seg", float(np.abs(got["seg_logits"] - ref_s["seg_logits"]).max()), flush=True)
print("DONE", OUT)
