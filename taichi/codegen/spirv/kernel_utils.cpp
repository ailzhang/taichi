#include "taichi/codegen/spirv/kernel_utils.h"

#include <unordered_map>

#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace spirv {

// static
std::string TaskAttributes::buffers_name(BufferInfo b) {
  if (b.type == BufferType::Args) {
    return "Args";
  }
  if (b.type == BufferType::Rets) {
    return "Rets";
  }
  if (b.type == BufferType::GlobalTmps) {
    return "GlobalTmps";
  }
  if (b.type == BufferType::Root) {
    return std::string("Root: ") + std::to_string(b.root_id);
  }
  TI_ERROR("unrecognized buffer type");
}

std::string TaskAttributes::debug_string() const {
  std::string result;
  result += fmt::format(
      "<TaskAttributes name={} advisory_total_num_threads={} "
      "task_type={} buffers=[ ",
      name, advisory_total_num_threads, offloaded_task_type_name(task_type));
  for (auto b : buffer_binds) {
    result += buffers_name(b.buffer) + " ";
  }
  result += "]";  // closes |buffers|
  // TODO(k-ye): show range_for
  result += ">";
  return result;
}

std::string TaskAttributes::BufferBind::debug_string() const {
  return fmt::format("<type={} binding={}>",
                     TaskAttributes::buffers_name(buffer), binding);
}

KernelContextAttributes::KernelContextAttributes(const Kernel &kernel)
    : args_bytes_(0),
      rets_bytes_(0),
      extra_args_bytes_(RuntimeContext::extra_args_size) {
  arg_attribs_vec_.reserve(kernel.args.size());
  for (const auto &ka : kernel.args) {
    ArgAttributes aa;
    aa.dtype = to_vk_dtype_enum(ka.dt);
    const size_t dt_bytes = data_type_size(ka.dt);
    aa.is_array = ka.is_array;
    if (aa.is_array) {
      aa.field_dim = ka.total_dim - ka.element_shape.size();
      aa.element_shape = ka.element_shape;
    }
    aa.stride = dt_bytes;
    aa.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(aa);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    size_t dt_bytes{0};
    if (auto tensor_type = kr.dt->cast<TensorType>()) {
      auto tensor_dtype = tensor_type->get_element_type();
      ra.dtype = to_vk_dtype_enum(tensor_dtype);
      dt_bytes = data_type_size(tensor_dtype);
      ra.is_array = true;
      ra.stride = tensor_type->get_num_elements() * dt_bytes;
    } else {
      ra.dtype = to_vk_dtype_enum(kr.dt);
      dt_bytes = data_type_size(kr.dt);
      ra.is_array = false;
      ra.stride = dt_bytes;
    }
    ra.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(ra);
  }

  auto arange_args = [](auto *vec, size_t offset, bool is_ret) -> size_t {
    size_t bytes = offset;
    for (int i = 0; i < vec->size(); ++i) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes =
          (attribs.is_array && !is_ret)
              ? sizeof(uint64_t)
              : data_type_size(vk_dtype_enum_to_dt(attribs.dtype));
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += is_ret ? attribs.stride : dt_bytes;
      TI_TRACE(
          "  at={} {} offset_in_mem={} stride={}",
          (*vec)[i].is_array ? (is_ret ? "array" : "vector ptr") : "scalar", i,
          attribs.offset_in_mem, attribs.stride);
    }
    return bytes - offset;
  };

  TI_TRACE("args:");
  args_bytes_ = arange_args(&arg_attribs_vec_, 0, false);
  // Align to extra args
  args_bytes_ = (args_bytes_ + 4 - 1) / 4 * 4;

  TI_TRACE("rets:");
  rets_bytes_ = arange_args(&ret_attribs_vec_, 0, true);

  TI_TRACE("sizes: args={} rets={}", args_bytes(), rets_bytes());
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

uint32_t to_vk_dtype_enum(DataType dt) {
  if (dt == PrimitiveType::u64) {
    return 0;
  } else if (dt == PrimitiveType::i64) {
    return 1;
  } else if (dt == PrimitiveType::u32) {
    return 2;
  } else if (dt == PrimitiveType::i32) {
    return 3;
  } else if (dt == PrimitiveType::u16) {
    return 4;
  } else if (dt == PrimitiveType::i16) {
    return 5;
  } else if (dt == PrimitiveType::u8) {
    return 6;
  } else if (dt == PrimitiveType::i8) {
    return 7;
  } else if (dt == PrimitiveType::f64) {
    return 8;
  } else if (dt == PrimitiveType::f32) {
    return 9;
  } else if (dt == PrimitiveType::f16) {
    return 10;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

DataType vk_dtype_enum_to_dt(uint32_t dt) {
  if (dt == 0) {
    return PrimitiveType::u64;
  } else if (dt == 1) {
    return PrimitiveType::i64;
  } else if (dt == 2) {
    return PrimitiveType::u32;
  } else if (dt == 3) {
    return PrimitiveType::i32;
  } else if (dt == 4) {
    return PrimitiveType::u16;
  } else if (dt == 5) {
    return PrimitiveType::i16;
  } else if (dt == 6) {
    return PrimitiveType::u8;
  } else if (dt == 7) {
    return PrimitiveType::i8;
  } else if (dt == 8) {
    return PrimitiveType::f64;
  } else if (dt == 9) {
    return PrimitiveType::f32;
  } else if (dt == 10) {
    return PrimitiveType::f16;
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
