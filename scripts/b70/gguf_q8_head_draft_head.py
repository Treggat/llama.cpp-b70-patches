#!/usr/bin/env python3
# Copy a GGUF, re-encoding only output.weight from BF16 to Q8_0. Every other tensor and all metadata are copied as-is.
# With a third argument N, also write blk.<n_layer>.nextn.draft_head = the first N rows of the Q8_0 head, which the
# Qwen3.5/3.8 MTP drafter uses instead of the full head (the target still verifies over the full vocab).
# Usage: gguf_q8_head_draft_head.py <src.gguf> <dst.gguf> [N]   e.g. N = 98304 covers ~99.95% of tokens for Qwen3.8
import sys
import numpy as np
import gguf

src, dst = sys.argv[1], sys.argv[2]
n_draft = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # also write blk.<n_layer>.nextn.draft_head = first n_draft rows
reader = gguf.GGUFReader(src)
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

Q8 = gguf.GGMLQuantizationType.Q8_0


def q8_output(t):
    raw = np.asarray(t.data)
    if raw.dtype == np.uint8:                     # the reader hands BF16 over as raw bytes
        raw = raw.view(np.uint16)
    rows, cols = raw.shape
    n_embd, n_vocab = int(t.shape[0]), int(t.shape[1])
    assert raw.dtype == np.uint16 and (rows, cols) == (n_vocab, n_embd), (raw.dtype, raw.shape, t.shape)
    bs, ts = gguf.GGML_QUANT_SIZES[Q8]
    out = np.empty((rows, cols // bs * ts), dtype=np.uint8)
    step = 8192
    for r0 in range(0, rows, step):
        f32 = (raw[r0:r0 + step].astype(np.uint32) << 16).view(np.float32)
        out[r0:r0 + step] = gguf.quants.quantize(f32, Q8)
        print(f"  output.weight rows {min(r0 + step, rows)}/{rows}", flush=True)
    return out


new_data = {}
for t in reader.tensors:
    if t.name == "output.weight":
        assert t.tensor_type == gguf.GGMLQuantizationType.BF16, t.tensor_type
        new_data[t.name] = q8_output(t)
        d = new_data[t.name]
        writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, Q8)
    else:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

dh_name = None
if n_draft > 0:
    n_layer = int(reader.fields[f"{arch}.block_count"].contents()) - int(reader.fields[f"{arch}.nextn_predict_layers"].contents())
    dh_name = f"blk.{n_layer}.nextn.draft_head.weight"
    assert dh_name not in [t.name for t in reader.tensors]
    new_data[dh_name] = np.ascontiguousarray(new_data["output.weight"][:n_draft])
    d = new_data[dh_name]
    writer.add_tensor_info(dh_name, d.shape, d.dtype, d.nbytes, Q8)
    print("adding", dh_name, d.shape, flush=True)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(new_data.get(t.name, t.data), tensor_endianess=reader.endianess)
if dh_name:
    writer.write_tensor_data(new_data[dh_name], tensor_endianess=reader.endianess)
writer.close()
print("done", dst, flush=True)
