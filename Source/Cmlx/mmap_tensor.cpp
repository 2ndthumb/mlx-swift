// mmap_tensor.cpp -- helper to build an MLX array from a memory-mapped file region.
#include "mlx/c/mmap.h"
#include "mlx/c/error.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <string>

extern "C" {

static mlx_array _tensor_from_data(const void* ptr,
                                   size_t offset,
                                   size_t bytes,
                                   const int* shape,
                                   int dim,
                                   mlx_dtype dtype) {
    // Use the public C API to create an array by *copying* the bytes for now.
    // In a future update we can expose an internal constructor that wraps the
    // buffer without copy.
    const char* base = static_cast<const char*>(ptr) + offset;
    return mlx_array_new_data(base, shape, dim, dtype);
}

mlx_array mlx_mmap_tensor(const char* path,
                          size_t offset,
                          const int* shape,
                          int dim,
                          mlx_dtype dtype,
                          mlx_device /*device*/) {
    size_t elementSize = mlx_dtype_size(dtype);
    size_t count = 1;
    for (int i = 0; i < dim; ++i) count *= static_cast<size_t>(shape[i]);
    size_t bytes = elementSize * count;

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        // return empty array on error
        return mlx_array_new();
    }

    void* ptr = ::mmap(nullptr, bytes + offset, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) {
        return mlx_array_new();
    }

    mlx_array arr = _tensor_from_data(ptr, offset, bytes, shape, dim, dtype);

    // We copied, so unmap immediately.
    ::munmap(ptr, bytes + offset);

    return arr;
}

} // extern "C" 