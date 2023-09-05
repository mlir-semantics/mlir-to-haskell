#include <optional>

// #include "llvm/IR/InstVisitor.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {

typedef llvm::raw_ostream stream_t; 
static std::string DIALECT_ATTR = "__hask_dialects";

struct ComputeDialectsPass
    : public PassWrapper<ComputeDialectsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComputeDialectsPass)

  StringRef getArgument() const final { return "compute-dialects"; }
  StringRef getDescription() const final { return "Computes the dialect used within each region, and save these as attributes."; }

  void visitOperation(std::optional<func::FuncOp> parentFuncOp, Operation *op) {
    // will add all "seen" dialects to the FuncOp's (parentFuncOp) attributes 
    // visit regions and blocks as usual, passing in the variable
    // if sees another FuncOp, though, 
    //  update the parameter to the funcOp seen, 
    //  then also add everything into the parent after regions are done 
    addDialect(parentFuncOp, op->getDialect()->getNamespace());

    // function: this guy needs dialect attributes saved
    std::optional<func::FuncOp> newParentFuncOp = parentFuncOp;
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      newParentFuncOp = funcOp;
    }

    for (Region &region : op->getRegions())
      visitRegion(newParentFuncOp, region);

    if (llvm::isa<func::FuncOp>(op)) {
      // assert newParentFuncOp and parentFuncOp is not equivalent
      // assert existence of attribute on parent op
      // addDialects(parentFuncOp, (*newParentFuncOp)->getDiscardableAttr(DIALECT_ATTR));
    }
  }

  void visitRegion(std::optional<func::FuncOp> parentFuncOp, Region &region) {
    for (Block &block : region.getBlocks())
      visitBlock(parentFuncOp, block);
  }

  void visitBlock(std::optional<func::FuncOp> parentFuncOp, Block &block) {
    for (Operation &op : block.getOperations())
      visitOperation(parentFuncOp, &op);
  }

  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    visitOperation(std::nullopt, op);
  }

private:
  void addDialect(std::optional<func::FuncOp> op, llvm::StringRef dialectName) {
    if (!op) return;

    // adds dialectName to the op's attributes
    if (!(*op)->hasAttr(DIALECT_ATTR)) {
      (*op)->setDiscardableAttr(
        StringAttr(DIALECT_ATTR), 
        DictionaryAttr::getWithSorted(op->getContext(), llvm::ArrayRef<NamedAttribute>())
      );
    }
  }

  void addDialects(std::optional<func::FuncOp> op, Attribute _attr) {
    assert(_attr && _attr.isa<DictionaryAttr>());
    if (!op) return;
    auto attr = _attr.dyn_cast<DictionaryAttr>();

    // adds dialects (stored in attr) to op.

  }
};
} // namespace

namespace mlir {
void registerComputeDialectsPass() {
  PassRegistration<ComputeDialectsPass>();
}
} // namespace mlir