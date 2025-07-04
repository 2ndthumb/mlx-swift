// MLXArray+MMap.swift
// Convenience memory-mapped initializer requested for KV off-loading support.
// This stub compiles on all platforms and captures the API surface.  A future
// revision will wire this into a zero-copy C++ buffer backed by `munmap`.

import Foundation
#if canImport(Darwin)
import Darwin
#endif
import Cmlx

extension MLXArray {
    /// Construct an `MLXArray` that views a memory-mapped file region.
    /// The returned array is *read-only* and assumes the underlying file
    /// remains valid for the lifetime of the array.
    ///
    /// - Parameters:
    ///   - fileURL: File on disk to map.
    ///   - offset: Byte offset inside the file where the tensor data starts.
    ///   - shape: Row-major shape of the tensor.
    ///   - dtype: Element data type.
    ///   - device: Target device for the final tensor (defaults to `.gpu(0)`).
    /// - Returns: A tensor backed by the mapped file region.
    ///
    /// **Implementation note**: This placeholder path still copies the data
    /// into an MLX buffer.  A follow-up C++ shim will replace this with a
    /// zero-copy `mlx::core::Buffer` that owns the `munmap` deleter.
    public static func mmap(
        fileURL: URL,
        offset: Int = 0,
        shape: [Int],
        dtype: DType,
        device: Device = .gpu
    ) throws -> MLXArray {
#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // Fast path: use C shim for zero-copy mmap if available.
        let shape32 = shape.map { Int32($0) }

        let arrCtx: mlx_array = shape32.withUnsafeBufferPointer { shpBuf in
            fileURL.path.withCString { cPath in
                // Call into C layer; if the symbol is missing (e.g. on non-Apple
                // builds) fallback to copy path below.
                if let fn = _optional_mlx_mmap_tensor {
                    return fn(cPath, numericCast(offset), shpBuf.baseAddress, Int32(shape32.count), dtype.cmlxDtype, device.ctx)
                } else {
                    return mlx_array_new() // placeholder, will copy fallback after loop
                }
            }
        }

        // If C shim returned a valid array use it; otherwise fallback to old copy-based path.
        if mlx_array_ndim(arrCtx) != 0 {
            return MLXArray(arrCtx)
        }

        // -------- Fallback (copy) --------
        // Calculate byte length.
        let elementSize = dtype.size
        let elementCount = shape.reduce(1, *)
        let byteCount = elementCount * elementSize

        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { close(fd) }

        let ptr = Darwin.mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, fd, off_t(offset))
        guard ptr != MAP_FAILED else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { Darwin.munmap(ptr, byteCount) }

        let copyCtx = shape32.withUnsafeBufferPointer { buf -> mlx_array in
            mlx_array_new_data(ptr, buf.baseAddress, Int32(shape32.count), dtype.cmlxDtype)
        }
        return MLXArray(copyCtx)
#else
        throw NSError(domain: "MLXArray", code: -1, userInfo: [NSLocalizedDescriptionKey: "Memory-mapped constructor only available on Apple platforms."])
#endif
    }
}

#if canImport(Darwin)
// Dynamically optional import of the C shim so that non-updated Cmlx builds
// continue to link.  We resolve it at runtime once.
private let _optional_mlx_mmap_tensor: (@convention(c) (UnsafePointer<CChar>, size_t, UnsafePointer<Int32>?, Int32, mlx_dtype, mlx_device) -> mlx_array)? = {
    guard let sym = dlsym(dlopen(nil, RTLD_LAZY), "mlx_mmap_tensor") else { return nil }
    typealias Fn = @convention(c) (UnsafePointer<CChar>, size_t, UnsafePointer<Int32>?, Int32, mlx_dtype, mlx_device) -> mlx_array
    return unsafeBitCast(sym, to: Fn?.self)
}()
#endif 