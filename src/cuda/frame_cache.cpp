#include "cuda/frame_cache.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace vnlbcu {
namespace {

[[noreturn]] void throw_cuda(cudaError_t status, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw_cuda(status, operation);
    }
}

class CudaFrameCacheBackend final : public FrameCacheBackend {
  public:
    explicit CudaFrameCacheBackend(int requested_device) {
        if (requested_device < 0) {
            check_cuda(cudaGetDevice(&device_), "cudaGetDevice(frame cache)");
        } else {
            device_ = requested_device;
            check_cuda(cudaSetDevice(device_), "cudaSetDevice(frame cache)");
        }
    }

    [[nodiscard]] void* allocate(std::size_t bytes) override {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(frame cache)");
        void* pointer = nullptr;
        check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc(frame cache slot)");
        return pointer;
    }

    void deallocate(void* pointer) noexcept override {
        if (pointer == nullptr) {
            return;
        }
        if (cudaSetDevice(device_) == cudaSuccess) {
            (void)cudaFree(pointer);
        }
    }

    [[nodiscard]] Event create_event() override {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(frame cache)");
        cudaEvent_t event = nullptr;
        check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "cudaEventCreateWithFlags(frame cache)");
        return reinterpret_cast<Event>(event);
    }

    void destroy_event(Event event) noexcept override {
        if (event == nullptr) {
            return;
        }
        if (cudaSetDevice(device_) == cudaSuccess) {
            (void)cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
        }
    }

    void record_event(Event event, cudaStream_t stream) override {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(frame cache)");
        check_cuda(
            cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), stream),
            "cudaEventRecord(frame cache)");
    }

    void wait_event(cudaStream_t stream, Event event) override {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(frame cache)");
        check_cuda(cudaStreamWaitEvent(stream,
                                       reinterpret_cast<cudaEvent_t>(event), 0),
                   "cudaStreamWaitEvent(frame cache)");
    }

  private:
    int device_ = 0;
};

void validate_keys(std::span<const FrameCacheKey> keys, std::size_t capacity) {
    if (keys.empty()) {
        throw std::invalid_argument(
            "CUDA frame-cache window must not be empty");
    }
    if (keys.size() > capacity) {
        throw std::invalid_argument(
            "CUDA frame-cache window exceeds the fixed slot count");
    }
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (keys[index].frame < 0) {
            throw std::invalid_argument(
                "CUDA frame-cache keys require non-negative frame indices");
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (keys[index] == keys[prior]) {
                throw std::invalid_argument(
                    "CUDA frame-cache windows require unique keys");
            }
        }
    }
}

} // namespace

std::shared_ptr<FrameCacheBackend> make_cuda_frame_cache_backend(int device) {
    return std::make_shared<CudaFrameCacheBackend>(device);
}

class FrameCache::Impl {
  public:
    struct ClaimResult {
        std::vector<CachedDeviceFrame> frames;
        std::vector<std::size_t> slots;
        std::vector<bool> loads;
    };

    Impl(std::size_t slot_count, std::size_t frame_bytes,
         std::shared_ptr<FrameCacheBackend> backend)
        : backend_(std::move(backend)), frame_bytes_(frame_bytes) {
        if (slot_count == 0 || frame_bytes_ == 0) {
            throw std::invalid_argument(
                "CUDA frame-cache slot count and frame size must be positive");
        }
        if (backend_ == nullptr) {
            throw std::invalid_argument("CUDA frame-cache backend is null");
        }

        slots_.resize(slot_count);
        try {
            for (Slot& slot : slots_) {
                slot.data = backend_->allocate(frame_bytes_);
                if (slot.data == nullptr) {
                    throw std::runtime_error(
                        "CUDA frame-cache backend returned a null allocation");
                }
                try {
                    slot.event = backend_->create_event();
                    if (slot.event == nullptr) {
                        throw std::runtime_error(
                            "CUDA frame-cache backend returned a null event");
                    }
                } catch (...) {
                    backend_->deallocate(slot.data);
                    slot.data = nullptr;
                    throw;
                }
            }
        } catch (...) {
            destroy_slots();
            throw;
        }
    }

    ~Impl() { destroy_slots(); }

    ClaimResult claim(std::span<const FrameCacheKey> keys,
                      bool defer_loading_hits) {
        validate_keys(keys, slots_.size());
        std::unique_lock lock(mutex_);

        while (true) {
            std::vector<std::size_t> selected(keys.size(), no_slot);
            bool waits_for_loading_key = false;

            for (std::size_t key_index = 0; key_index < keys.size();
                 ++key_index) {
                const std::size_t found = find_key_locked(keys[key_index]);
                if (found == no_slot) {
                    continue;
                }
                if (slots_[found].state == FrameCacheSlotState::Loading &&
                    !defer_loading_hits) {
                    waits_for_loading_key = true;
                    break;
                }
                selected[key_index] = found;
            }

            if (waits_for_loading_key) {
                ++stats_.waits;
                condition_.wait(lock);
                continue;
            }

            std::vector<bool> hit_slot(slots_.size(), false);
            std::size_t missing = 0;
            for (std::size_t key_index = 0; key_index < keys.size();
                 ++key_index) {
                if (selected[key_index] == no_slot) {
                    ++missing;
                } else {
                    hit_slot[selected[key_index]] = true;
                }
            }

            std::vector<std::size_t> victims;
            victims.reserve(slots_.size());
            for (std::size_t slot_index = 0; slot_index < slots_.size();
                 ++slot_index) {
                const Slot& slot = slots_[slot_index];
                if (slot.references != 0 ||
                    slot.state == FrameCacheSlotState::Loading ||
                    hit_slot[slot_index]) {
                    continue;
                }
                victims.push_back(slot_index);
            }
            std::sort(victims.begin(), victims.end(),
                      [&](std::size_t left, std::size_t right) {
                          const Slot& lhs = slots_[left];
                          const Slot& rhs = slots_[right];
                          if (lhs.state != rhs.state) {
                              return lhs.state == FrameCacheSlotState::Empty;
                          }
                          return lhs.stamp < rhs.stamp;
                      });

            if (victims.size() < missing) {
                // No references or key assignments have changed.  Waiting
                // while holding zero partial claims is the central deadlock
                // avoidance invariant of this cache.
                ++stats_.waits;
                condition_.wait(lock);
                continue;
            }

            std::size_t victim_cursor = 0;
            ClaimResult result;
            result.frames.resize(keys.size());
            result.slots.resize(keys.size());
            result.loads.resize(keys.size());

            for (std::size_t key_index = 0; key_index < keys.size();
                 ++key_index) {
                std::size_t slot_index = selected[key_index];
                if (slot_index == no_slot) {
                    slot_index = victims[victim_cursor++];
                    result.loads[key_index] = true;
                }
                result.slots[key_index] = slot_index;
                result.frames[key_index] = CachedDeviceFrame{
                    .key = keys[key_index],
                    .slot = slot_index,
                    .data = slots_[slot_index].data,
                    .bytes = frame_bytes_,
                    .needs_upload = result.loads[key_index],
                };
                if (slots_[slot_index].references ==
                    std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error(
                        "CUDA frame-cache slot reference count overflow");
                }
            }

            // Any allocation needed to renormalize the LRU clock happens
            // before slot state is changed, preserving the all-or-nothing
            // claim guarantee even in the exceptional wraparound path.
            prepare_stamps_locked(keys.size());

            for (std::size_t key_index = 0; key_index < keys.size();
                 ++key_index) {
                const std::size_t slot_index = result.slots[key_index];
                const bool load = result.loads[key_index];
                Slot& slot = slots_[slot_index];
                if (load) {
                    Slot& victim = slot;
                    if (victim.state == FrameCacheSlotState::Ready) {
                        ++stats_.evictions;
                    }
                    victim.key = keys[key_index];
                    victim.state = FrameCacheSlotState::Loading;
                    ++stats_.misses;
                } else {
                    ++stats_.hits;
                }

                ++slot.references;
                slot.stamp = next_stamp_locked();
            }
            return result;
        }
    }

    void record_load_events(const std::vector<std::size_t>& slots,
                            const std::vector<bool>& loads,
                            cudaStream_t stream) {
        for (std::size_t index = 0; index < slots.size(); ++index) {
            if (loads[index]) {
                backend_->record_event(slots_[slots[index]].event, stream);
            }
        }
    }

    void publish(const std::vector<CachedDeviceFrame>& frames,
                 const std::vector<std::size_t>& slots,
                 const std::vector<bool>& loads) {
        {
            std::lock_guard lock(mutex_);
            for (std::size_t index = 0; index < slots.size(); ++index) {
                if (!loads[index]) {
                    continue;
                }
                const Slot& slot = slots_[slots[index]];
                if (slot.state != FrameCacheSlotState::Loading ||
                    !slot.key.has_value() ||
                    slot.key.value() != frames[index].key ||
                    slot.references == 0) {
                    throw std::logic_error("CUDA frame-cache loading claim "
                                           "changed before publish");
                }
            }
            for (std::size_t index = 0; index < slots.size(); ++index) {
                if (loads[index]) {
                    slots_[slots[index]].state = FrameCacheSlotState::Ready;
                }
            }
        }
        condition_.notify_all();
    }

    void wait_hits(const std::vector<CachedDeviceFrame>& frames,
                   const std::vector<std::size_t>& slots,
                   const std::vector<bool>& loads, cudaStream_t stream) {
        wait_selected(frames, slots, loads, false, stream);
    }

    void wait_ready(const std::vector<CachedDeviceFrame>& frames,
                    const std::vector<std::size_t>& slots,
                    const std::vector<bool>& loads, cudaStream_t stream) {
        wait_selected(frames, slots, loads, true, stream);
    }

    void abandon(const std::vector<CachedDeviceFrame>& frames,
                 const std::vector<std::size_t>& slots,
                 const std::vector<bool>& loads) noexcept {
        {
            std::lock_guard lock(mutex_);
            abandon_locked(frames, slots, loads);
        }
        condition_.notify_all();
    }

    void release(const std::vector<CachedDeviceFrame>& frames,
                 const std::vector<std::size_t>& slots,
                 const std::vector<bool>& loads,
                 bool abandon_loading) noexcept {
        {
            std::lock_guard lock(mutex_);
            if (abandon_loading) {
                abandon_locked(frames, slots, loads);
            }
            for (const std::size_t slot_index : slots) {
                Slot& slot = slots_[slot_index];
                if (slot.references == 0) {
                    std::terminate();
                }
                --slot.references;
            }
        }
        condition_.notify_all();
    }

    [[nodiscard]] std::size_t slot_count() const noexcept {
        return slots_.size();
    }

    [[nodiscard]] std::size_t frame_bytes() const noexcept {
        return frame_bytes_;
    }

    [[nodiscard]] FrameCacheStats stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    [[nodiscard]] std::vector<FrameCacheSlotSnapshot> snapshot() const {
        std::lock_guard lock(mutex_);
        std::vector<FrameCacheSlotSnapshot> result;
        result.reserve(slots_.size());
        for (const Slot& slot : slots_) {
            result.push_back(FrameCacheSlotSnapshot{
                .state = slot.state,
                .key = slot.key,
                .references = slot.references,
                .lru_stamp = slot.stamp,
            });
        }
        return result;
    }

  private:
    struct Slot {
        std::optional<FrameCacheKey> key{};
        void* data = nullptr;
        FrameCacheBackend::Event event = nullptr;
        std::uint32_t references = 0;
        std::uint64_t stamp = 0;
        FrameCacheSlotState state = FrameCacheSlotState::Empty;
    };

    static constexpr std::size_t no_slot =
        std::numeric_limits<std::size_t>::max();

    [[nodiscard]] std::size_t
    find_key_locked(const FrameCacheKey& key) const noexcept {
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            if (slots_[index].state != FrameCacheSlotState::Empty &&
                slots_[index].key.has_value() &&
                slots_[index].key.value() == key) {
                return index;
            }
        }
        return no_slot;
    }

    void prepare_stamps_locked(std::size_t count) {
        const auto max_stamp = std::numeric_limits<std::uint64_t>::max();
        if (count > max_stamp) {
            throw std::overflow_error(
                "CUDA frame-cache LRU request is too large");
        }
        const auto required = static_cast<std::uint64_t>(count);
        if (clock_ > max_stamp - required) {
            std::vector<std::size_t> order(slots_.size());
            for (std::size_t index = 0; index < order.size(); ++index) {
                order[index] = index;
            }
            std::sort(order.begin(), order.end(),
                      [&](std::size_t left, std::size_t right) {
                          return slots_[left].stamp < slots_[right].stamp;
                      });
            clock_ = 0;
            for (const std::size_t index : order) {
                slots_[index].stamp = ++clock_;
            }
        }
    }

    [[nodiscard]] std::uint64_t next_stamp_locked() noexcept {
        return ++clock_;
    }

    void wait_selected(const std::vector<CachedDeviceFrame>& frames,
                       const std::vector<std::size_t>& slots,
                       const std::vector<bool>& loads, bool include_loads,
                       cudaStream_t stream) {
        std::vector<FrameCacheBackend::Event> events;
        events.reserve(slots.size());
        {
            std::unique_lock lock(mutex_);
            const auto resolved = [&]() noexcept {
                for (std::size_t index = 0; index < slots.size(); ++index) {
                    if (!include_loads && loads[index]) {
                        continue;
                    }
                    const Slot& slot = slots_[slots[index]];
                    if (slot.state == FrameCacheSlotState::Loading &&
                        slot.key.has_value() &&
                        slot.key.value() == frames[index].key) {
                        return false;
                    }
                }
                return true;
            };
            if (!resolved()) {
                ++stats_.waits;
                condition_.wait(lock, resolved);
            }

            for (std::size_t index = 0; index < slots.size(); ++index) {
                if (!include_loads && loads[index]) {
                    continue;
                }
                const Slot& slot = slots_[slots[index]];
                if (slot.state != FrameCacheSlotState::Ready ||
                    !slot.key.has_value() ||
                    slot.key.value() != frames[index].key) {
                    throw std::runtime_error(
                        "CUDA frame-cache producer abandoned a deferred hit");
                }
                events.push_back(slot.event);
            }
        }

        // Lease references prevent reuse after the state lock is released.
        for (const FrameCacheBackend::Event event : events) {
            backend_->wait_event(stream, event);
        }
    }

    void abandon_locked(const std::vector<CachedDeviceFrame>& frames,
                        const std::vector<std::size_t>& slots,
                        const std::vector<bool>& loads) noexcept {
        for (std::size_t index = 0; index < slots.size(); ++index) {
            if (!loads[index]) {
                continue;
            }
            Slot& slot = slots_[slots[index]];
            if (slot.state == FrameCacheSlotState::Loading &&
                slot.key.has_value() && slot.key.value() == frames[index].key) {
                slot.state = FrameCacheSlotState::Empty;
                slot.key.reset();
            }
        }
    }

    void destroy_slots() noexcept {
        if (backend_ == nullptr) {
            return;
        }
        for (Slot& slot : slots_) {
            backend_->destroy_event(slot.event);
            slot.event = nullptr;
            backend_->deallocate(slot.data);
            slot.data = nullptr;
        }
    }

    std::shared_ptr<FrameCacheBackend> backend_;
    std::size_t frame_bytes_ = 0;
    std::vector<Slot> slots_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::uint64_t clock_ = 0;
    FrameCacheStats stats_{};
};

FrameCache::Window::Window(std::shared_ptr<Impl> impl,
                           std::vector<CachedDeviceFrame> frames,
                           std::vector<std::size_t> slots,
                           std::vector<bool> loads) noexcept
    : impl_(std::move(impl)), frames_(std::move(frames)),
      slots_(std::move(slots)), loads_(std::move(loads)),
      published_(std::none_of(loads_.begin(), loads_.end(),
                              [](bool load) { return load; })) {}

FrameCache::Window::~Window() { release(); }

FrameCache::Window::Window(Window&& other) noexcept
    : impl_(std::move(other.impl_)), frames_(std::move(other.frames_)),
      slots_(std::move(other.slots_)), loads_(std::move(other.loads_)),
      published_(std::exchange(other.published_, false)),
      abandoned_(std::exchange(other.abandoned_, false)) {}

FrameCache::Window& FrameCache::Window::operator=(Window&& other) noexcept {
    if (this != &other) {
        release();
        impl_ = std::move(other.impl_);
        frames_ = std::move(other.frames_);
        slots_ = std::move(other.slots_);
        loads_ = std::move(other.loads_);
        published_ = std::exchange(other.published_, false);
        abandoned_ = std::exchange(other.abandoned_, false);
    }
    return *this;
}

std::span<const CachedDeviceFrame> FrameCache::Window::frames() const noexcept {
    return frames_;
}

bool FrameCache::Window::valid() const noexcept { return impl_ != nullptr; }

bool FrameCache::Window::needs_upload() const noexcept {
    return std::any_of(
        frames_.begin(), frames_.end(),
        [](const CachedDeviceFrame& frame) { return frame.needs_upload; });
}

bool FrameCache::Window::published() const noexcept { return published_; }

void FrameCache::Window::publish(cudaStream_t stream) {
    if (!valid()) {
        throw std::logic_error("CUDA frame-cache window is not valid");
    }
    if (abandoned_) {
        throw std::logic_error("CUDA frame-cache window was abandoned");
    }
    if (published_) {
        return;
    }

    // Recording every event first makes the subsequent state transition
    // atomic from other acquirers' perspective.  A partial record failure
    // leaves all keys Loading; release() will abandon the whole set.
    impl_->record_load_events(slots_, loads_, stream);
    impl_->publish(frames_, slots_, loads_);
    published_ = true;
}

void FrameCache::Window::wait_ready(cudaStream_t stream) const {
    if (!valid()) {
        throw std::logic_error("CUDA frame-cache window is not valid");
    }
    if (!published_) {
        throw std::logic_error(
            "CUDA frame-cache window must be published before waiting");
    }
    impl_->wait_ready(frames_, slots_, loads_, stream);
}

void FrameCache::Window::wait_hits(cudaStream_t stream) const {
    if (!valid()) {
        throw std::logic_error("CUDA frame-cache window is not valid");
    }
    impl_->wait_hits(frames_, slots_, loads_, stream);
}

void FrameCache::Window::abandon() noexcept {
    if (!valid() || published_ || abandoned_) {
        return;
    }
    impl_->abandon(frames_, slots_, loads_);
    abandoned_ = true;
}

void FrameCache::Window::release() noexcept {
    if (!valid()) {
        return;
    }
    impl_->release(frames_, slots_, loads_, !published_ && !abandoned_);
    impl_.reset();
    frames_.clear();
    slots_.clear();
    loads_.clear();
    published_ = false;
    abandoned_ = false;
}

FrameCache::FrameCache(std::size_t slot_count, std::size_t frame_bytes,
                       int device)
    : FrameCache(slot_count, frame_bytes,
                 make_cuda_frame_cache_backend(device)) {}

FrameCache::FrameCache(std::size_t slot_count, std::size_t frame_bytes,
                       std::shared_ptr<FrameCacheBackend> backend)
    : impl_(std::make_shared<Impl>(slot_count, frame_bytes,
                                   std::move(backend))) {}

FrameCache::~FrameCache() = default;

FrameCache::FrameCache(FrameCache&&) noexcept = default;

FrameCache& FrameCache::operator=(FrameCache&&) noexcept = default;

FrameCache::Window FrameCache::acquire(std::span<const FrameCacheKey> keys) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA FrameCache was moved from");
    }
    const std::shared_ptr<Impl> impl = impl_;
    Impl::ClaimResult claim = impl->claim(keys, false);
    return Window(impl, std::move(claim.frames), std::move(claim.slots),
                  std::move(claim.loads));
}

FrameCache::Window
FrameCache::acquire_deferred(std::span<const FrameCacheKey> keys) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA FrameCache was moved from");
    }
    const std::shared_ptr<Impl> impl = impl_;
    Impl::ClaimResult claim = impl->claim(keys, true);
    return Window(impl, std::move(claim.frames), std::move(claim.slots),
                  std::move(claim.loads));
}

std::size_t FrameCache::slot_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->slot_count();
}

std::size_t FrameCache::frame_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->frame_bytes();
}

FrameCacheStats FrameCache::stats() const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA FrameCache was moved from");
    }
    return impl_->stats();
}

std::vector<FrameCacheSlotSnapshot> FrameCache::snapshot() const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA FrameCache was moved from");
    }
    return impl_->snapshot();
}

} // namespace vnlbcu
