#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <array>
#include <atomic>
#include <thread>

/**
 * Lock‑free ring buffer for passing frames between producer and consumer threads.
 * Stores std::shared_ptr<cv::Mat> to avoid copies. Overwrites the oldest frame
 * when the buffer is full.
 */
template <size_t Capacity = 64>
class FrameBufferLockFree {
public:
    FrameBufferLockFree() : head_(0), tail_(0) {}

    // Push a new frame with its id onto the buffer. Overwrites oldest if full.
    void push(const std::shared_ptr<cv::Mat> &frame, int frame_id) {
        size_t pos = tail_.load(std::memory_order_relaxed);
        size_t next = (pos + 1) % Capacity;
        // If buffer is full, advance head to discard the oldest frame.
        if (next == head_.load(std::memory_order_acquire)) {
            head_.store((head_.load(std::memory_order_relaxed) + 1) % Capacity,
                        std::memory_order_release);
        }
        frames_[pos] = frame;
        ids_[pos] = frame_id;
        tail_.store(next, std::memory_order_release);
    }

    // Try to pop a frame without blocking. Returns false if buffer empty.
    bool try_pop(std::pair<std::shared_ptr<cv::Mat>, int> &out) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_acquire);
        if (h == t) {
            return false; // empty
        }
        out.first = frames_[h];
        out.second = ids_[h];
        head_.store((h + 1) % Capacity, std::memory_order_release);
        return true;
    }

    // Blocking pop for backward compatibility – spins until a frame is available.
    std::pair<std::shared_ptr<cv::Mat>, int> pop() {
        std::pair<std::shared_ptr<cv::Mat>, int> out;
        while (!try_pop(out)) {
            std::this_thread::yield();
        }
        return out;
    }

    // Clear the buffer.
    void clear() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    std::array<std::shared_ptr<cv::Mat>, Capacity> frames_{};
    std::array<int, Capacity> ids_{};
    std::atomic<size_t> head_{};
    std::atomic<size_t> tail_{};
};
