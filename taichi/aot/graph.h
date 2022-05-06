#pragma once
#include "taichi/aot/module_loader.h"
#include <unordered_set>
namespace taichi {
namespace lang {
namespace aot {
  class Kernel;
  class Graph;
  struct Arg {
    std::string name;
  };
    struct IValue {
    public:
    enum Tag {SCALAR, DEVALLOC};
    uint64 val;
    Tag tag;
    uint64 size_;
    std::vector<int> shape_; // Hack! remove me

    IValue(const DeviceAllocation& devalloc, uint64 size, const std::vector<int>& shape) :
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
  class Node {
    public:
    // use void for now, it might return value to host in the future. 
    Node() = default;
    virtual ~Node() = default;
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
    Node(Node &&) = default;
    Node &operator=(Node &&) = default;
    virtual void run(taichi::lang::RuntimeContext* host_ctx, std::unordered_map<std::string, IValue>& args) = 0;
  };
  class Dispatch : public Node {
    public:
    explicit Dispatch(Kernel *kernel, const std::vector<Arg>& args): kernel_(kernel), symbolic_args_(args) {}
    void run(taichi::lang::RuntimeContext* host_ctx, std::unordered_map<std::string, IValue>& args) override;
    private:
    Kernel *kernel_{nullptr};
    std::vector<Arg> symbolic_args_;
  };

  class Sequential : public Node {
    public:
    explicit Sequential(Graph* graph):owning_graph_(graph) {}
    void append(Node* node);
    void emplace(Kernel* kernel, const std::vector<Arg>& args);
    void run(taichi::lang::RuntimeContext* host_ctx, std::unordered_map<std::string, IValue>& args) override;

    private:
      std::vector<Node*> sequence_;
      Graph* owning_graph_{nullptr};
  };

  class Graph {
    public:
    Graph();
    void bind();
    void run(std::unordered_map<std::string, IValue>& args);
    Node* create_dispatch(Kernel* kernel, const std::vector<Arg>& args);
    Node* create_sequential();
    Sequential* seq();

    ~Graph() {
      for (const Node* n: all_nodes_) {
        delete n;
      }
    }
    
    private:
    std::unique_ptr<Sequential> seq_{nullptr};

    std::unordered_set<const Node*> all_nodes_;
    std::unordered_map<std::string, IValue> bound_args;
  };
}
}
}