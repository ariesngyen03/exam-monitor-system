#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <utility>

// Simple thread‑safe buffer for passing frames between producer and consumer threads.
// Stores a shared_ptr to cv::Mat to avoid unnecessary copies.
class FrameBuffer {
public:
    // Push a new frame with its id onto the buffer.
    void push(const std::shared_ptr<cv::Mat>& frame, int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.emplace(frame, frame_id);
        cond_.notify_one();
    }

    // Pop a frame; blocks if the queue is empty.
    std::pair<std::shared_ptr<cv::Mat>, int> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        auto item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Try to pop without blocking; returns false if empty.
    bool try_pop(std::pair<std::shared_ptr<cv::Mat>, int>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Clear the buffer.
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

private:
    std::queue<std::pair<std::shared_ptr<cv::Mat>, int>> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};
