#pragma once

template <typename T>
inline void append(std::vector<T>& lhs, std::vector<T>& rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}
