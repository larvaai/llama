#pragma once

#include <chrono>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace model_worker {

class PendingCancelRegistry {
public:
    using Clock = std::chrono::steady_clock;
    using Key = std::pair<std::string, std::string>;

    explicit PendingCancelRegistry(
        std::size_t max_entries = 1024,
        Clock::duration ttl = std::chrono::minutes(5)
    );

    void add(const Key & key, Clock::time_point now = Clock::now());
    bool consume(const Key & key, Clock::time_point now = Clock::now());
    std::size_t size(Clock::time_point now = Clock::now());

private:
    void expire(Clock::time_point now);

    std::size_t max_entries_;
    Clock::duration ttl_;
    std::map<Key, Clock::time_point> entries_;
};

}  // namespace model_worker
