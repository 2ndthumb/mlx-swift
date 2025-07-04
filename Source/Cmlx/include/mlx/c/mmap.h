/*
 * mmap.h -- zero-copy (or fallback copy) support for mapping on-disk tensors
 * into MLX arrays.
 */

#ifndef MLX_C_MMAP_H
#define MLX_C_MMAP_H

#include "mlx/c/array.h"
#include "mlx/c/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Map a file region as an MLX array.
///
/// - path: UTF-8 path to the file to map.
/// - offset: byte offset where tensor data starts.
/// - shape: pointer to `dim` integers describing row-major shape.
/// - dim: number of dimensions.
/// - dtype: element dtype.
/// - device: destination device (cpu/gpu).
///
/// Returns: a *new* mlx_array (caller owns and must free).
mlx_array mlx_mmap_tensor(const char* path,
                          size_t offset,
                          const int* shape,
                          int dim,
                          mlx_dtype dtype,
                          mlx_device device);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* MLX_C_MMAP_H */ 