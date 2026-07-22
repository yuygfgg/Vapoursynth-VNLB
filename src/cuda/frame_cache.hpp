#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace vnlbcu {

// source_id identifies a producer/node, while generation separates cache
// layouts or content revisions belonging to that producer.  One slot stores a
// complete, uniformly sized device frame for this key.
struct FrameCacheKey {
    std::uint64_t source_id = 0;
    int frame = 0;
    std::uint32_t generation = 0;

    friend constexpr bool operator==(const FrameCacheKey&,
                                     const FrameCacheKey&) noexcept = default;
};

// CUDA operations are injected so the ownership/concurrency state machine can
// be tested without a GPU.  Implementations must be thread-safe; deallocation
// and event destruction must not throw.
class FrameCacheBackend {
  public:
    using Event = void*;

    virtual ~FrameCacheBackend() = default;
    [[nodiscard]] virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* pointer) noexcept = 0;
    [[nodiscard]] virtual Event create_event() = 0;
    virtual void destroy_event(Event event) noexcept = 0;
    virtual void record_event(Event event, cudaStream_t stream) = 0;
    virtual void wait_event(cudaStream_t stream, Event event) = 0;
};

[[nodiscard]] std::shared_ptr<FrameCacheBackend>
make_cuda_frame_cache_backend(int device = -1);

enum class FrameCacheSlotState : std::uint8_t {
    Empty,
    Loading,
    Ready,
};

struct FrameCacheSlotSnapshot {
    FrameCacheSlotState state = FrameCacheSlotState::Empty;
    std::optional<FrameCacheKey> key{};
    std::uint32_t references = 0;
    std::uint64_t lru_stamp = 0;
};

struct FrameCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t waits = 0;
};

struct CachedDeviceFrame {
    FrameCacheKey key{};
    std::size_t slot = 0;
    void* data = nullptr;
    std::size_t bytes = 0;
    bool needs_upload = false;
};

// Fixed-capacity, refcounted device frame cache.  acquire() reserves an entire
// window atomically: a waiter never holds a partial set of slots, so two
// overlapping windows cannot deadlock while each waits for the other's
// remaining slots.  A loading window must be published or abandoned; the RAII
// lease abandons unpublished slots automatically on error.
//
// The caller must finish all stream work which reads a lease's slots before
// release/destruction.  Once references reach zero an LRU replacement may
// reuse the device memory and re-record its ready event.
class FrameCache final {
  private:
    class Impl;

  public:
    class Window final {
      public:
        Window() noexcept = default;
        ~Window();

        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;
        Window(Window&& other) noexcept;
        Window& operator=(Window&& other) noexcept;

        [[nodiscard]] std::span<const CachedDeviceFrame>
        frames() const noexcept;
        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] bool needs_upload() const noexcept;
        [[nodiscard]] bool published() const noexcept;

        // Record every missing slot's ready event on stream, then publish all
        // of them under one cache lock.  No slot becomes acquirable in Loading
        // state.  If event recording throws, the window remains unpublished
        // and must be abandoned after any work writing those slots is done.
        void publish(cudaStream_t stream);

        // Enqueue waits only for entries which were Ready when this window was
        // acquired.  Missing entries belong exclusively to this lease and can
        // therefore be written and consumed on the same stream before they are
        // published.  This supports a single final status synchronization:
        // wait for hits, produce/consume misses, check asynchronous status,
        // then publish on success or abandon on failure.
        void wait_hits(cudaStream_t stream) const;

        // Enqueue waits for every frame's recorded ready event.  This is valid
        // immediately for an all-hit window, or after publish() for a window
        // containing misses.
        void wait_ready(cudaStream_t stream) const;

        // Remove every still-loading key atomically.  Existing hit references
        // remain held until release().  The caller must first ensure no stream
        // work can still read or write a removed loading slot.
        void abandon() noexcept;

        // Abandon if necessary, decrement every slot reference, and invalidate
        // the lease.  Stream work using these pointers must already be done.
        void release() noexcept;

      private:
        friend class FrameCache;
        Window(std::shared_ptr<Impl> impl,
               std::vector<CachedDeviceFrame> frames,
               std::vector<std::size_t> slots,
               std::vector<bool> loads) noexcept;

        std::shared_ptr<Impl> impl_;
        std::vector<CachedDeviceFrame> frames_;
        std::vector<std::size_t> slots_;
        std::vector<bool> loads_;
        bool published_ = false;
        bool abandoned_ = false;
    };

    FrameCache(std::size_t slot_count, std::size_t frame_bytes,
               int device = -1);
    FrameCache(std::size_t slot_count, std::size_t frame_bytes,
               std::shared_ptr<FrameCacheBackend> backend);
    ~FrameCache();

    FrameCache(const FrameCache&) = delete;
    FrameCache& operator=(const FrameCache&) = delete;
    FrameCache(FrameCache&&) noexcept;
    FrameCache& operator=(FrameCache&&) noexcept;

    // Blocks until every key can be claimed together.  Keys must be unique,
    // non-negative frame indices and the window may not exceed slot_count().
    [[nodiscard]] Window acquire(std::span<const FrameCacheKey> keys);

    [[nodiscard]] std::size_t slot_count() const noexcept;
    [[nodiscard]] std::size_t frame_bytes() const noexcept;
    [[nodiscard]] FrameCacheStats stats() const;
    [[nodiscard]] std::vector<FrameCacheSlotSnapshot> snapshot() const;

  private:
    std::shared_ptr<Impl> impl_;
};

} // namespace vnlbcu
