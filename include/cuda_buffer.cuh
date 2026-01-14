// cuda_buffer.h
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

// Helper macro for checking CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("[CUDA Error] " + std::string(cudaGetErrorString(err))); \
    } \
} while (0)

template <typename T>
class CudaBuffer {
public:
    size_t count;
    size_t size_bytes;

    // Default constructor
    CudaBuffer() : d_ptr(nullptr), count(0), size_bytes(0) {}
    
    // Constructor: allocates GPU memory
    CudaBuffer(size_t count) : d_ptr(nullptr), count(count), size_bytes(count * sizeof(T)) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc((void**)&d_ptr, size_bytes));
        }
    }

    // Destructor: frees GPU memory
    ~CudaBuffer() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }

    // Disable copying to prevent double-free errors
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Implement move semantics for efficient transfers of ownership
    CudaBuffer(CudaBuffer&& other) noexcept : d_ptr(other.d_ptr), count(other.count), size_bytes(other.size_bytes) {
        other.d_ptr = nullptr;
        other.count = 0;
        other.size_bytes = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        CudaBuffer temp(std::move(other));
        swap(temp);
        return *this;
    }

    // Copy data from a host vector to the device
    void to_device(const std::vector<T>& host_vec) {
        if (host_vec.size() != count) throw std::runtime_error("Size mismatch in to_device");
        CUDA_CHECK(cudaMemcpy(d_ptr, host_vec.data(), size_bytes, cudaMemcpyHostToDevice));
    }
    
    // Copy data from the device to a host vector
    void from_device(std::vector<T>& host_vec) {
        if (host_vec.size() != count) host_vec.resize(count);
        CUDA_CHECK(cudaMemcpy(host_vec.data(), d_ptr, size_bytes, cudaMemcpyDeviceToHost));
    }

    void swap(CudaBuffer<T>& other) noexcept {
        std::swap(d_ptr, other.d_ptr);
        std::swap(count, other.count);
        std::swap(size_bytes, other.size_bytes);
    }

    void resize(size_t new_count, bool clear_memory = false) {
        if (new_count == count) {
            if (clear_memory && d_ptr) {
                CUDA_CHECK(cudaMemset(d_ptr, 0, size_bytes));
            }
            return;
        }

        // Free old memory
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
        }

        count = new_count;
        size_bytes = new_count * sizeof(T);

        // Allocate new memory
        if (new_count > 0) {
            CUDA_CHECK(cudaMalloc((void**)&d_ptr, size_bytes));
            if (clear_memory) {
                CUDA_CHECK(cudaMemset(d_ptr, 0, size_bytes));
            }
        }
    }

    void clear() {
        if (d_ptr && size_bytes > 0) {
            CUDA_CHECK(cudaMemset(d_ptr, 0, size_bytes));
        }
    }
    
    // Get the raw device pointer to pass to kernels
    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }

private:
    T* d_ptr;

};

