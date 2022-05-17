#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"
#include "tests/cpp/program/test_program.h"
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#endif

using namespace taichi;
using namespace lang;

[[maybe_unused]] static void aot_save() {
  auto program = Program(Arch::vulkan);

  program.config.advanced_optimization = false;

  int n = 10;

  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->dense(Axis(0), n, false);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/true);

  auto aot_builder = program.make_aot_module_builder(Arch::vulkan);

  std::unique_ptr<Kernel> kernel_init, kernel_ret, kernel_simple_ret;

  {
    /*
    @ti.kernel
    def ret() -> ti.f32:
      sum = 0.2
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::f32);
    builder.create_local_store(sum, builder.get_float32(0.2));
    builder.create_return(builder.create_local_load(sum));

    kernel_simple_ret =
        std::make_unique<Kernel>(program, builder.extract_ir(), "simple_ret");
    kernel_simple_ret->insert_ret(PrimitiveType::f32);
  }

  {
    /*
    @ti.kernel
    def init():
      for index in range(n):
        place[index] = index
    */
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, index);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  {
    /*
    @ti.kernel
    def ret():
      sum = 0
      for index in place:
        sum = sum + place[index];
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::i32);
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *sum_old = builder.create_local_load(sum);
      auto *place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");
    kernel_ret->insert_ret(PrimitiveType::i32);
  }

  aot_builder->add("simple_ret", kernel_simple_ret.get());
  aot_builder->add_field("place", place, true, place->dt, {n}, 1, 1);
  aot_builder->add("init", kernel_init.get());
  aot_builder->add("ret", kernel_ret.get());
  aot_builder->dump(".", "");
}

[[maybe_unused]] static void save_ndarray_kernels(Arch arch) {
  TestProgram test_prog;
  test_prog.setup(arch);
  auto aot_builder = test_prog.prog()->make_aot_module_builder(arch);
  IRBuilder builder1, builder2;

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
  auto ker1 =
      std::make_unique<Kernel>(*test_prog.prog(), std::move(block), "ker1");
  ker1->insert_arg(get_data_type<int>(), /*is_array=*/true);
  {
    auto *arg0 = builder2.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                          /*is_ptr=*/true);
    auto *arg1 = builder2.create_arg_load(/*arg_id=*/1, get_data_type<int>(),
                                          /*is_ptr=*/false);
    auto *one = builder2.get_int32(1);
    auto *a1ptr = builder2.create_external_ptr(arg0, {one});
    builder2.create_global_store(a1ptr, arg1);  // a[1] = arg1
  }
  auto block2 = builder2.extract_ir();
  auto ker2 =
      std::make_unique<Kernel>(*test_prog.prog(), std::move(block2), "ker2");
  ker2->insert_arg(get_data_type<int>(), /*is_array=*/true);
  ker2->insert_arg(get_data_type<int>(), /*is_array=*/false);

  aot_builder->add("ker1", ker1.get());
  aot_builder->add("ker2", ker2.get());
  aot_builder->dump(".", "");
}

#ifdef TI_WITH_VULKAN
[[maybe_unused]] static void write_devalloc(
    taichi::lang::vulkan::VkRuntime *vulkan_runtime,
    taichi::lang::DeviceAllocation &alloc,
    const void *data,
    size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  std::memcpy(device_arr_ptr, data, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}

[[maybe_unused]] static void load_devalloc(
    taichi::lang::vulkan::VkRuntime *vulkan_runtime,
    taichi::lang::DeviceAllocation &alloc,
    void *data,
    size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  std::memcpy(data, device_arr_ptr, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}

TEST(AotSaveLoad, Vulkan) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  aot_save();

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version =
      taichi::lang::vulkan::VulkanEnvSettings::kApiVersion();
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // Run AOT module loader
  vulkan::AotModuleParams mod_params;
  mod_params.module_path = ".";
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 64);
  vulkan_runtime->add_root_buffer(root_size);

  auto simple_ret_kernel = vk_module->get_kernel("simple_ret");
  EXPECT_TRUE(simple_ret_kernel);

  simple_ret_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();
  EXPECT_FLOAT_EQ(host_ctx.get_ret<float>(0), 0.2);

  auto init_kernel = vk_module->get_kernel("init");
  EXPECT_TRUE(init_kernel);

  auto ret_kernel = vk_module->get_kernel("ret");
  EXPECT_TRUE(ret_kernel);

  auto ret2_kernel = vk_module->get_kernel("ret2");
  EXPECT_FALSE(ret2_kernel);

  // Run kernels
  init_kernel->launch(&host_ctx);
  ret_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();

  // Retrieve data
  auto x_field = vk_module->get_field("place");
  EXPECT_NE(x_field, nullptr);
}

TEST(AotSaveLoad, VulkanNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  save_ndarray_kernels(Arch::vulkan);

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version =
      taichi::lang::vulkan::VulkanEnvSettings::kApiVersion();
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // Run AOT module loader
  vulkan::AotModuleParams mod_params;
  mod_params.module_path = ".";
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  vulkan_runtime->add_root_buffer(root_size);

  auto ker1 = vk_module->get_kernel("ker1");
  EXPECT_TRUE(ker1);

  const int size = 10;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ =
      embedded_device->device()->allocate_memory(alloc_params);
  host_ctx.set_arg_devalloc(0, devalloc_arr_, {10});

  int src[size] = {0};
  src[0] = 2;
  src[2] = 40;
  write_devalloc(vulkan_runtime.get(), devalloc_arr_, src, sizeof(src));
  ker1->launch(&host_ctx);
  vulkan_runtime->synchronize();

  int dst[size] = {33};
  load_devalloc(vulkan_runtime.get(), devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 1);
  EXPECT_EQ(dst[2], 42);

  auto ker2 = vk_module->get_kernel("ker2");
  EXPECT_TRUE(ker2);

  host_ctx.set_arg(1, 3);
  ker2->launch(&host_ctx);
  vulkan_runtime->synchronize();
  load_devalloc(vulkan_runtime.get(), devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 3);
  EXPECT_EQ(dst[2], 42);

  // Deallocate
  embedded_device->device()->dealloc_memory(devalloc_arr_);
}
#endif
