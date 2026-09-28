#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    // [switchcost] prompt-cache delta save (LLAMA_PCACHE_KEEP): an optional host copy of an earlier state of the same
    // sequence. A memory that knows which of its rows are unchanged since that copy was taken writes them with
    // write_ref(): `src` points into the reference and holds the same bytes as tensor[offset, offset + size).
    // The default falls back to reading the tensor.
    virtual const uint8_t * get_ref(size_t & size) const { size = 0; return nullptr; }
    virtual void write_ref(const uint8_t * /*src*/, ggml_tensor * tensor, size_t offset, size_t size) {
        write_tensor(tensor, offset, size);
    }

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
