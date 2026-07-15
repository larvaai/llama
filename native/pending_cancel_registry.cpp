#include "pending_cancel_registry.h"

#include <algorithm>

namespace model_worker {

PendingCancelRegistry::PendingCancelRegistry(std::size_t max_entries, Clock::duration ttl)
    : max_entries_(max_entries), ttl_(ttl) {
    if (max_entries_ == 0 || ttl_ <= Clock::duration::zero()) {
        throw std::invalid_argument("pending cancel registry requires positive bounds");
    }
}

void PendingCancelRegistry::expire(Clock::time_point now) {
    for (auto entry = entries_.begin(); entry != entries_.end();) {
        if (entry->second <= now) {
            entry = entries_.erase(entry);
        } else {
            ++entry;
        }
    }
}

void PendingCancelRegistry::add(const Key & key, Clock::time_point now) {
    if (key.first.empty() || key.second.empty()) {
        return;
    }
    expire(now);
    auto existing = entries_.find(key);
    if (existing != entries_.end()) {
        existing->second = now + ttl_;
        return;
    }
    if (entries_.size() >= max_entries_) {
        const auto oldest = std::min_element(
            entries_.begin(),
            entries_.end(),
            [](const auto & left, const auto & right) { return left.second < right.second; }
        );
        entries_.erase(oldest);
    }
    entries_.emplace(key, now + ttl_);
}

bool PendingCancelRegistry::consume(const Key & key, Clock::time_point now) {
    expire(now);
    const auto entry = entries_.find(key);
    if (entry == entries_.end()) {
        return false;
    }
    entries_.erase(entry);
    return true;
}

std::size_t PendingCancelRegistry::size(Clock::time_point now) {
    expire(now);
    return entries_.size();
}

}  // namespace model_worker
