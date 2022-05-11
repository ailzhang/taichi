#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"
#include "tests/cpp/program/test_program.h"
#include "taichi/program/graph_module.h"

using namespace taichi;
using namespace lang;

[[maybe_unused]] static std::unique_ptr<Kernel> setup_kernel1(Program* prog) {
  IRBuilder builder1;
  {
  auto *arg = builder1.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                      /*is_ptr=*/true);
  auto *zero = builder1.get_int32(0);
  auto *one = builder1.get_int32(1);
  auto *two = builder1.get_int32(2);
  auto *a1ptr = builder1.create_external_ptr(arg, {one});
  builder1.create_global_store(a1ptr, one);  // a[1] = 1
  auto *a0 =
      builder1.create_global_load(builder1.create_external_ptr(arg, {zero}));
  auto *a2ptr = builder1.create_external_ptr(arg, {two});
  auto *a2 = builder1.create_global_load(a2ptr);
  auto *a0plusa2 = builder1.create_add(a0, a2);
  builder1.create_global_store(a2ptr, a0plusa2);  // a[2] = a[0] + a[2]
  }
  auto block = builder1.extract_ir();
  auto ker1 = std::make_unique<Kernel>(*prog, std::move(block), "ker1");
  ker1->insert_arg(get_data_type<int>(), /*is_array=*/true);
  return ker1;
}

[[maybe_unused]] static std::unique_ptr<Kernel> setup_kernel2(Program* prog) {
  IRBuilder builder2;

  {
  auto *arg = builder2.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                      /*is_ptr=*/true);
  auto *arg2 = builder2.create_arg_load(/*arg_id=*/1, get_data_type<int>(),
                                      /*is_ptr=*/false);
  auto *one = builder2.get_int32(1);
  auto *a1ptr = builder2.create_external_ptr(arg, {one});
  builder2.create_global_store(a1ptr, arg2);  // a[1] = 2
  }
  auto block2 = builder2.extract_ir();
  auto ker2 = std::make_unique<Kernel>(*prog, std::move(block2), "ker2");
  ker2->insert_arg(get_data_type<int>(), /*is_array=*/true);
  ker2->insert_arg(get_data_type<int>(), /*is_array=*/false);
  return ker2;
}

TEST(GraphModule, SimpleGraphRun) {
  TestProgram test_prog;
  // FIXME: Change this to x64 before sending a PR
  test_prog.setup(Arch::vulkan);
  const int size = 10;

  auto ker1 = setup_kernel1(test_prog.prog());
  auto ker2 = setup_kernel2(test_prog.prog());

  auto g = std::make_unique<Graph>("test");
  auto seq = g->seq();
  auto arr_arg = Arg{"arr"};
  seq->emplace(ker1.get(), {arr_arg});
  seq->emplace(ker2.get(), {arr_arg, Arg{"x"}});
  g->compile();

  auto array = Ndarray(test_prog.prog(), PrimitiveType::i32, {size});
  array.write_int({0}, 2);
  array.write_int({2}, 40);
  std::unordered_map<std::string, IValue> args;
  args.insert({"arr", IValue(array)});
  args.insert({"x", IValue::create<int>(2)});

  g->run(args);
  EXPECT_EQ(array.read_int({0}), 2);
  EXPECT_EQ(array.read_int({1}), 2);
  EXPECT_EQ(array.read_int({2}), 42);
}
