#include "cuda/frame_cache.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(message) +
                                 ": wrong exception: " + error.what());
    }
    throw std::runtime_error(std::string(message) + ": no exception");
}

template <typename Predicate>
bool eventually(Predicate&& predicate,
                std::chrono::milliseconds timeout = 2000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (std::forward<Predicate>(predicate)()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return std::forward<Predicate>(predicate)();
}

cudaStream_t fake_stream(std::uintptr_t id) noexcept {
    return reinterpret_cast<cudaStream_t>(id);
}

vnlbcu::FrameCacheKey key(int frame, std::uint64_t source_id = 1,
                          std::uint32_t generation = 0) {
    return vnlbcu::FrameCacheKey{
        .source_id = source_id,
        .frame = frame,
        .generation = generation,
    };
}

class FakeBackend final : public vnlbcu::FrameCacheBackend {
  private:
    struct FakeEvent {
        std::size_t id = 0;
    };

  public:
    [[nodiscard]] void* allocate(std::size_t bytes) override {
        void* pointer = ::operator new(bytes);
        ++live_allocations_;
        return pointer;
    }

    void deallocate(void* pointer) noexcept override {
        if (pointer != nullptr) {
            ::operator delete(pointer);
            --live_allocations_;
        }
    }

    [[nodiscard]] Event create_event() override {
        const std::size_t call = ++create_calls_;
        if (fail_create_call_.load() == call) {
            throw std::runtime_error("injected create_event failure");
        }
        auto* event = new FakeEvent{.id = ++next_event_id_};
        ++live_events_;
        return event;
    }

    void destroy_event(Event event) noexcept override {
        if (event != nullptr) {
            delete static_cast<FakeEvent*>(event);
            --live_events_;
        }
    }

    void record_event(Event event, cudaStream_t stream) override {
        const std::size_t call = ++record_calls_;
        if (fail_record_call_.load() == call) {
            throw std::runtime_error("injected record_event failure");
        }
        const auto* fake_event = static_cast<const FakeEvent*>(event);
        std::lock_guard lock(activity_mutex_);
        records_.push_back(Activity{fake_event->id, stream});
    }

    void wait_event(cudaStream_t stream, Event event) override {
        const auto* fake_event = static_cast<const FakeEvent*>(event);
        ++wait_calls_;
        std::lock_guard lock(activity_mutex_);
        waits_.push_back(Activity{fake_event->id, stream});
    }

    void fail_create_on(std::size_t call) noexcept { fail_create_call_ = call; }

    void fail_record_on(std::size_t call) noexcept { fail_record_call_ = call; }

    void reset_activity() {
        std::lock_guard lock(activity_mutex_);
        records_.clear();
        waits_.clear();
        record_calls_ = 0;
        wait_calls_ = 0;
        fail_record_call_ = 0;
    }

    [[nodiscard]] std::size_t record_count() const noexcept {
        return record_calls_.load();
    }

    [[nodiscard]] std::size_t wait_count() const noexcept {
        return wait_calls_.load();
    }

    [[nodiscard]] std::size_t live_allocations() const noexcept {
        return live_allocations_.load();
    }

    [[nodiscard]] std::size_t live_events() const noexcept {
        return live_events_.load();
    }

  private:
    struct Activity {
        std::size_t event_id = 0;
        cudaStream_t stream = nullptr;
    };

    std::atomic_size_t live_allocations_{0};
    std::atomic_size_t live_events_{0};
    std::atomic_size_t next_event_id_{0};
    std::atomic_size_t create_calls_{0};
    std::atomic_size_t record_calls_{0};
    std::atomic_size_t wait_calls_{0};
    std::atomic_size_t fail_create_call_{0};
    std::atomic_size_t fail_record_call_{0};
    mutable std::mutex activity_mutex_;
    std::vector<Activity> records_;
    std::vector<Activity> waits_;
};

const vnlbcu::FrameCacheSlotSnapshot*
find_slot(std::span<const vnlbcu::FrameCacheSlotSnapshot> slots,
          const vnlbcu::FrameCacheKey& requested) {
    for (const auto& slot : slots) {
        if (slot.key.has_value() && slot.key.value() == requested) {
            return &slot;
        }
    }
    return nullptr;
}

void seed(vnlbcu::FrameCache& cache,
          std::span<const vnlbcu::FrameCacheKey> keys) {
    auto window = cache.acquire(keys);
    require(window.needs_upload(), "seed window unexpectedly contained hits");
    window.publish(fake_stream(1));
    window.release();
}

void test_constructor_cleanup() {
    auto backend = std::make_shared<FakeBackend>();
    backend->fail_create_on(2);
    require_throws<std::runtime_error>(
        [&] { vnlbcu::FrameCache cache(3, 256, backend); },
        "constructor event failure");
    require(backend->live_allocations() == 0,
            "constructor failure leaked allocations");
    require(backend->live_events() == 0, "constructor failure leaked events");

    require_throws<std::invalid_argument>(
        [&] { vnlbcu::FrameCache cache(0, 256, backend); }, "zero slot count");
    require_throws<std::invalid_argument>(
        [&] { vnlbcu::FrameCache cache(1, 0, backend); }, "zero frame size");
    require_throws<std::invalid_argument>(
        [&] {
            vnlbcu::FrameCache cache(
                1, 256, std::shared_ptr<vnlbcu::FrameCacheBackend>{});
        },
        "null backend");
}

void test_miss_publish_hit_and_lifetime() {
    auto backend = std::make_shared<FakeBackend>();
    std::optional<vnlbcu::FrameCache::Window> surviving_window;

    {
        vnlbcu::FrameCache cache(2, 128, backend);
        const std::array keys{key(3), key(4)};
        auto first = cache.acquire(keys);
        require(first.valid() && first.needs_upload() && !first.published(),
                "first window must contain unpublished misses");
        require(first.frames().size() == keys.size(),
                "first window size mismatch");
        require(first.frames()[0].data != first.frames()[1].data,
                "cache slots must own distinct allocations");

        const std::array<void*, 2> pointers{first.frames()[0].data,
                                            first.frames()[1].data};
        first.wait_hits(fake_stream(2));
        require(backend->wait_count() == 0,
                "all-miss wait_hits enqueued an event wait");
        require_throws<std::logic_error>(
            [&] { first.wait_ready(fake_stream(2)); },
            "wait_ready before publish");

        first.publish(fake_stream(1));
        require(first.published(), "publish did not update lease state");
        require(backend->record_count() == 2,
                "publish did not record every miss event");
        first.wait_ready(fake_stream(2));
        require(backend->wait_count() == 2,
                "wait_ready did not wait for both frames");
        first.release();

        backend->reset_activity();
        auto second = cache.acquire(keys);
        require(second.published() && !second.needs_upload(),
                "published keys were not cache hits");
        require(second.frames()[0].data == pointers[0] &&
                    second.frames()[1].data == pointers[1],
                "cache hit did not preserve slot pointers");
        second.wait_hits(fake_stream(3));
        require(backend->wait_count() == 2,
                "all-hit wait_hits did not wait for both events");
        second.publish(fake_stream(3));
        require(backend->record_count() == 0,
                "publishing an all-hit lease recorded events");
        second.wait_ready(fake_stream(3));
        require(backend->wait_count() == 4,
                "all-hit wait_ready did not wait for both events");

        const auto stats = cache.stats();
        require(stats.misses == 2 && stats.hits == 2,
                "cache hit/miss statistics mismatch");
        surviving_window.emplace(std::move(second));
    }

    require(backend->live_allocations() == 2 && backend->live_events() == 2,
            "active lease did not retain cache storage");
    surviving_window->release();
    surviving_window.reset();
    require(backend->live_allocations() == 0 && backend->live_events() == 0,
            "last lease release did not destroy cache storage");
}

void test_mixed_window_delayed_publish() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(3, 64, backend);
    const std::array initial{key(10)};
    seed(cache, initial);
    backend->reset_activity();

    const std::array requested{key(10), key(11)};
    auto window = cache.acquire(requested);
    require(!window.frames()[0].needs_upload && window.frames()[1].needs_upload,
            "mixed window hit/miss classification mismatch");
    window.wait_hits(fake_stream(5));
    require(backend->wait_count() == 1, "wait_hits must skip the loading slot");
    require(backend->record_count() == 0,
            "wait_hits unexpectedly published a loading slot");

    const auto loading_snapshot = cache.snapshot();
    const auto* miss = find_slot(loading_snapshot, key(11));
    require(miss != nullptr &&
                miss->state == vnlbcu::FrameCacheSlotState::Loading,
            "delayed miss was visible as Ready before publish");

    window.publish(fake_stream(5));
    require(backend->record_count() == 1,
            "mixed publish recorded the wrong event count");
    window.release();

    const std::array verify_keys{key(11)};
    auto verify = cache.acquire(verify_keys);
    require(!verify.needs_upload(), "delayed publish did not create a hit");
}

void test_lru_eviction() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(2, 64, backend);
    const std::array initial{key(20), key(21)};

    auto first = cache.acquire(initial);
    void* const old_second_pointer = first.frames()[1].data;
    const std::size_t old_second_slot = first.frames()[1].slot;
    first.publish(fake_stream(1));
    first.release();

    const std::array touch_key{key(20)};
    auto touch = cache.acquire(touch_key);
    require(!touch.needs_upload(), "LRU touch was not a hit");
    touch.release();

    const std::array replacement_key{key(22)};
    auto replacement = cache.acquire(replacement_key);
    require(replacement.needs_upload(), "replacement was not a miss");
    require(replacement.frames()[0].slot == old_second_slot &&
                replacement.frames()[0].data == old_second_pointer,
            "LRU did not reuse the least recently used slot");
    replacement.publish(fake_stream(1));
    replacement.release();

    const auto snapshot = cache.snapshot();
    require(find_slot(snapshot, key(20)) != nullptr,
            "recently used key was evicted");
    require(find_slot(snapshot, key(21)) == nullptr,
            "least recently used key survived eviction");
    require(find_slot(snapshot, key(22)) != nullptr,
            "replacement key is absent");
    require(cache.stats().evictions == 1, "eviction statistic mismatch");
}

void test_key_validation() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(2, 64, backend);

    require_throws<std::invalid_argument>(
        [&] { (void)cache.acquire(std::span<const vnlbcu::FrameCacheKey>{}); },
        "empty window");
    const std::array too_many{key(1), key(2), key(3)};
    require_throws<std::invalid_argument>(
        [&] { (void)cache.acquire(too_many); }, "oversized window");
    const std::array duplicate{key(1), key(1)};
    require_throws<std::invalid_argument>(
        [&] { (void)cache.acquire(duplicate); }, "duplicate key");
    const std::array negative{key(-1)};
    require_throws<std::invalid_argument>(
        [&] { (void)cache.acquire(negative); }, "negative frame");

    const auto snapshot = cache.snapshot();
    for (const auto& slot : snapshot) {
        require(slot.state == vnlbcu::FrameCacheSlotState::Empty &&
                    slot.references == 0 && !slot.key.has_value(),
                "invalid request changed cache state");
    }

    const std::array distinct{
        key(7, 1, 0),
        key(7, 2, 0),
    };
    auto valid = cache.acquire(distinct);
    require(valid.needs_upload(),
            "source_id did not distinguish otherwise equal keys");
}

void test_loading_key_waits_for_publish() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(1, 64, backend);
    const std::array requested{key(30)};
    auto owner = cache.acquire(requested);

    const std::uint64_t waits_before = cache.stats().waits;
    auto future = std::async(std::launch::async, [&cache, requested] {
        return cache.acquire(requested);
    });
    const bool reached_wait =
        eventually([&] { return cache.stats().waits > waits_before; });
    const bool remained_blocked =
        future.wait_for(20ms) == std::future_status::timeout;

    owner.publish(fake_stream(1));
    auto follower = future.get();
    require(reached_wait && remained_blocked,
            "acquire did not wait for a matching Loading key");
    require(!follower.needs_upload(),
            "waiter did not observe the published key as a hit");
    const auto snapshot = cache.snapshot();
    const auto* slot = find_slot(snapshot, requested[0]);
    require(slot != nullptr && slot->references == 2,
            "published key reference count mismatch");
    follower.release();
    owner.release();
}

void test_deferred_loading_hit_waits_at_consumption() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(2, 64, backend);
    const std::array owner_keys{key(32)};
    auto owner = cache.acquire(owner_keys);

    const std::array follower_keys{key(32), key(33)};
    const std::uint64_t waits_before = cache.stats().waits;
    auto follower = cache.acquire_deferred(follower_keys);
    require(!follower.frames()[0].needs_upload &&
                follower.frames()[1].needs_upload,
            "deferred window did not retain a Loading hit and claim its miss");
    require(cache.stats().waits == waits_before,
            "deferred acquire blocked on a Loading hit");

    auto future = std::async(std::launch::async, [&follower] {
        follower.wait_hits(fake_stream(2));
    });
    const bool reached_wait =
        eventually([&] { return cache.stats().waits > waits_before; });
    const bool remained_blocked =
        future.wait_for(20ms) == std::future_status::timeout;

    owner.publish(fake_stream(1));
    future.get();
    require(reached_wait && remained_blocked,
            "deferred hit was consumed before its producer published");
    require(backend->wait_count() == 1,
            "deferred hit did not enqueue its producer event wait");

    follower.publish(fake_stream(2));
    follower.release();
    owner.release();
}

void test_deferred_loading_hit_reports_abandon() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(1, 64, backend);
    const std::array requested{key(34)};
    auto owner = cache.acquire(requested);
    auto follower = cache.acquire_deferred(requested);

    const std::uint64_t waits_before = cache.stats().waits;
    auto future = std::async(std::launch::async, [&follower] {
        follower.wait_hits(fake_stream(2));
    });
    require(eventually([&] { return cache.stats().waits > waits_before; }),
            "deferred hit did not wait for its producer");

    owner.abandon();
    owner.release();
    require_throws<std::runtime_error>([&] { future.get(); },
                                       "deferred producer abandon");
    follower.release();
    require(cache.snapshot()[0].references == 0,
            "abandoned deferred hit leaked a reference");
}

void test_no_partial_claim_while_waiting_for_victim() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(2, 64, backend);
    const std::array initial{key(40), key(41)};
    seed(cache, initial);

    const std::array first_key{key(40)};
    const std::array second_key{key(41)};
    auto hold_first = cache.acquire(first_key);
    auto hold_second = cache.acquire(second_key);

    const std::array requested{key(40), key(42)};
    const std::uint64_t waits_before = cache.stats().waits;
    auto future = std::async(std::launch::async, [&cache, requested] {
        return cache.acquire(requested);
    });
    const bool reached_wait =
        eventually([&] { return cache.stats().waits > waits_before; });
    const bool remained_blocked =
        future.wait_for(20ms) == std::future_status::timeout;
    const auto blocked_snapshot = cache.snapshot();
    const auto* first = find_slot(blocked_snapshot, key(40));
    const auto* second = find_slot(blocked_snapshot, key(41));
    const bool no_partial_claim =
        first != nullptr && first->references == 1 && second != nullptr &&
        second->references == 1 &&
        find_slot(blocked_snapshot, key(42)) == nullptr;

    hold_second.release();
    auto waiter = future.get();
    require(reached_wait && remained_blocked,
            "window did not wait when every victim was referenced");
    require(no_partial_claim,
            "waiting window retained a hit or installed a partial miss");
    require(!waiter.frames()[0].needs_upload && waiter.frames()[1].needs_upload,
            "unblocked window hit/miss classification mismatch");
    waiter.publish(fake_stream(1));
    waiter.release();
    hold_first.release();
}

void test_abandon_and_raii_wake_waiters() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(1, 64, backend);
    const std::array requested{key(50)};
    std::optional<vnlbcu::FrameCache::Window> owner;
    owner.emplace(cache.acquire(requested));

    const std::uint64_t waits_before = cache.stats().waits;
    auto future = std::async(std::launch::async, [&cache, requested] {
        return cache.acquire(requested);
    });
    const bool reached_wait =
        eventually([&] { return cache.stats().waits > waits_before; });
    owner.reset();
    auto follower = future.get();
    require(reached_wait, "waiter did not block on abandoned Loading key");
    require(follower.needs_upload(), "RAII abandon left a poisoned cache hit");
    follower.publish(fake_stream(1));
    follower.release();

    const std::array another{key(51)};
    auto explicit_abandon = cache.acquire(another);
    explicit_abandon.abandon();
    const auto abandoned_snapshot = cache.snapshot();
    require(abandoned_snapshot[0].state == vnlbcu::FrameCacheSlotState::Empty &&
                abandoned_snapshot[0].references == 1 &&
                !abandoned_snapshot[0].key.has_value(),
            "explicit abandon did not atomically remove the loading key");
    explicit_abandon.release();
    require(cache.snapshot()[0].references == 0,
            "release after abandon leaked a reference");
}

void test_record_failure_rolls_back_whole_window() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(2, 64, backend);
    const std::array requested{key(60), key(61)};
    auto window = cache.acquire(requested);
    backend->fail_record_on(2);

    require_throws<std::runtime_error>([&] { window.publish(fake_stream(1)); },
                                       "injected event record failure");
    require(!window.published(),
            "partially recorded window was marked published");
    const auto loading_snapshot = cache.snapshot();
    for (const auto& slot : loading_snapshot) {
        require(slot.state == vnlbcu::FrameCacheSlotState::Loading &&
                    slot.references == 1,
                "record failure partially published the window");
    }

    window.release();
    const auto abandoned_snapshot = cache.snapshot();
    for (const auto& slot : abandoned_snapshot) {
        require(slot.state == vnlbcu::FrameCacheSlotState::Empty &&
                    slot.references == 0 && !slot.key.has_value(),
                "record failure left a poisoned Loading slot");
    }

    backend->reset_activity();
    auto retry = cache.acquire(requested);
    require(retry.needs_upload(),
            "record failure rollback produced a false cache hit");
    retry.publish(fake_stream(1));
    retry.release();
}

void test_multithreaded_stress() {
    auto backend = std::make_shared<FakeBackend>();
    vnlbcu::FrameCache cache(4, 64, backend);
    std::atomic_bool failed{false};
    std::mutex failure_mutex;
    std::string failure_message;
    std::vector<std::thread> threads;

    constexpr int thread_count = 6;
    constexpr int iterations = 400;
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        threads.emplace_back([&, thread_index] {
            try {
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    const int first = (iteration + thread_index * 3) % 8;
                    int second =
                        (first + 1 + ((iteration + thread_index) % 5)) % 8;
                    if (second == first) {
                        second = (second + 1) % 8;
                    }
                    const std::array requested{key(first), key(second)};
                    auto window = cache.acquire(requested);
                    window.wait_hits(fake_stream(
                        static_cast<std::uintptr_t>(thread_index + 1)));
                    if (window.needs_upload()) {
                        window.publish(fake_stream(
                            static_cast<std::uintptr_t>(thread_index + 1)));
                    }
                    window.wait_ready(fake_stream(
                        static_cast<std::uintptr_t>(thread_index + 1)));
                    window.release();
                }
            } catch (const std::exception& error) {
                failed = true;
                std::lock_guard lock(failure_mutex);
                if (failure_message.empty()) {
                    failure_message = error.what();
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    require(!failed.load(),
            std::string("multithreaded cache failure: ") + failure_message);
    const auto snapshot = cache.snapshot();
    for (const auto& slot : snapshot) {
        require(slot.state != vnlbcu::FrameCacheSlotState::Loading,
                "stress test left a Loading slot");
        require(slot.references == 0, "stress test leaked a slot reference");
    }
    require(cache.stats().waits > 0, "stress test never exercised a wait path");
}

} // namespace

int main() {
    try {
        test_constructor_cleanup();
        test_miss_publish_hit_and_lifetime();
        test_mixed_window_delayed_publish();
        test_lru_eviction();
        test_key_validation();
        test_loading_key_waits_for_publish();
        test_deferred_loading_hit_waits_at_consumption();
        test_deferred_loading_hit_reports_abandon();
        test_no_partial_claim_while_waiting_for_victim();
        test_abandon_and_raii_wake_waiters();
        test_record_failure_rolls_back_whole_window();
        test_multithreaded_stress();
        std::cout << "CUDA frame-cache CPU tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA frame-cache CPU test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
