#!/usr/bin/env python3
# Copy a GGUF, keeping only the first N rows of the MTP draft head blk.<n_layer>.nextn.draft_head.weight (any type:
# quantized rows are independent, so a row slice of the raw bytes is exactly the first N rows). Every other tensor and
# all metadata are copied as-is. The drafter then proposes only ids < N; the target still verifies over the full vocab.
# Usage: gguf_draft_head_trim.py <src.gguf> <dst.gguf> <N>     e.g. N = 65536 (dh64k) or 49152 (dh48k)
import sys
import numpy as np
import gguf

src, dst, n_keep = sys.argv[1], sys.argv[2], int(sys.argv[3])
reader = gguf.GGUFReader(src)
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

n_layer = int(reader.fields[f"{arch}.block_count"].contents()) - int(reader.fields[f"{arch}.nextn_predict_layers"].contents())
dh_name = f"blk.{n_layer}.nextn.draft_head.weight"

new_data = {}
found = False
for t in reader.tensors:
    data = t.data
    if t.name == dh_name:
        found = True
        n_rows = int(t.shape[1])
        assert 0 < n_keep < n_rows, (n_keep, n_rows)
        assert data.shape[0] == n_rows, (data.shape, t.shape)   # one row of raw bytes (or elements) per vocab id
        data = np.ascontiguousarray(data[:n_keep])
        new_data[t.name] = data
        print(f"{dh_name}: {gguf.GGMLQuantizationType(t.tensor_type).name} rows {n_rows} -> {n_keep}", flush=True)
    writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
assert found, f"{dh_name} not in {src}"

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for i, t in enumerate(reader.tensors):
    writer.write_tensor_data(new_data.get(t.name, t.data), tensor_endianess=reader.endianess)
    if i % 100 == 0:
        print(f"  tensor {i}/{len(reader.tensors)}", flush=True)
writer.close()
print("done", dst, flush=True)
