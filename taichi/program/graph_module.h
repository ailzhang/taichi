#pragma once

#include <string>
#include <vector>
#include <unordered_set>

// #include "taichi/program/context.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/aot/module_data.h"


namespace taichi {
namespace lang {
  class Kernel;
  class Graph;
  struct IValue;
  class AotModuleBuilder;
  struct Arg {
    std::string name;
  };

  class Node {
    public:
    // use void for now, it might return value to host in the future. 
    Node() = default;
    virtual ~Node() = default;
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
    Node(Node &&) = default;
    Node &operator=(Node &&) = default;
    virtual void compile() = 0;
    virtual void run(std::unordered_map<std::string, IValue>& args) = 0;
    virtual std::vector<aot::SymbolicDispatch> serialize(AotModuleBuilder* aot_builder) const = 0;
  };

  struct IValue {
    public:
    enum Tag {SCALAR, NDARRAY, DEVALLOC};
    uint64 val;
    Tag tag;
    uint64 size_;
    std::vector<int64> shape_; // Hack! remove me
    IValue(const Ndarray& ndarray) :
      IValue(reinterpret_cast<intptr_t>(&ndarray), Tag::NDARRAY) {}
    IValue(const DeviceAllocation& devalloc, uint64 size, const std::vector<int64>& shape) :
      IValue(reinterpret_cast<intptr_t>(&devalloc), Tag::DEVALLOC) {
        size_ = size;
        shape_ = shape;
      }
    template <typename T>
    static IValue create(T v) {
      return IValue(taichi_union_cast_with_different_sizes<uint64>(v), Tag::SCALAR);
    } 
    private:
    IValue(uint64 val, Tag tag): val(val), tag(tag) {}
  };

  class Dispatch : public Node {
    public:
    explicit Dispatch(Kernel *kernel, const std::vector<Arg>& args): kernel_(kernel), symbolic_args_(args) {}
    void compile() override;
    void run(std::unordered_map<std::string, IValue>& args) override;
    std::vector<aot::SymbolicDispatch> serialize(AotModuleBuilder* aot_builder) const override;
    private:
    Kernel *kernel_{nullptr};
    // std::string name;
    std::vector<Arg> symbolic_args_;
  };

  class Sequential : public Node {
    public:
    explicit Sequential(Graph* graph):owning_graph_(graph) {}
    void append(Node* node);
    void emplace(Kernel* kernel, const std::vector<Arg>& args);
    void compile() override;
    void run(std::unordered_map<std::string, IValue>& args) override;
    std::vector<aot::SymbolicDispatch> serialize(AotModuleBuilder* aot_builder) const override;

    private:
      std::vector<Node*> sequence_;
      Graph* owning_graph_{nullptr};
  };

  enum NodeKind {DISPATCH, SEQUENTIAL, COND};

  class Graph {
    public:
    Graph(std::string name);
    void compile();
    void bind();
    void run(std::unordered_map<std::string, IValue>& args);
    Node* create_dispatch(Kernel* kernel, const std::vector<Arg>& args);
    Node* create_sequential();
    Sequential* seq() const;
    void serialize(AotModuleBuilder* aot_builder) const;
    ~Graph() {
      for (const Node* n: all_nodes_) {
        delete n;
      }
    }
    
    private:
    std::string name_;
    std::unique_ptr<Sequential> seq_{nullptr};
    std::unordered_set<const Node*> all_nodes_;
    std::unordered_map<std::string, IValue> bound_args;
  };

}
}