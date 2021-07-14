#include "gtest/gtest.h"

#include "taichi/csrc/common/core.h"

namespace taichi {

TEST(CoreTest, Basic) {
  EXPECT_EQ(trim_string("hello taichi  "), "hello taichi");
}

}  // namespace taichi
