/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/csrc/common/core.h"
#include "taichi/csrc/common/task.h"
#include "taichi/csrc/util/testing.h"

TI_NAMESPACE_BEGIN

class RunTests : public Task {
  virtual std::string run(const std::vector<std::string> &parameters) {
    return std::to_string(run_tests(parameters));
  }
};

TI_IMPLEMENTATION(Task, RunTests, "test");

TI_NAMESPACE_END
