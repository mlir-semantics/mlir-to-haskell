#include <optional>
#include <set>

// #include "llvm/IR/InstVisitor.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {

typedef std::string DialectName;
typedef std::set<std::string> DialectsSet;
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
    addDialect(parentFuncOp, op->getDialect()->getNamespace().str());

    // function: this guy needs dialect attributes saved
    std::optional<func::FuncOp> newParentFuncOp = parentFuncOp;
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      newParentFuncOp = funcOp;
    }

    for (Region &region : op->getRegions())
      visitRegion(newParentFuncOp, region);

    if (llvm::isa<func::FuncOp>(op)) {
      assert(newParentFuncOp != parentFuncOp);
      addDialects(parentFuncOp, dialectsOf[*newParentFuncOp]);
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

    for (auto it = dialectsOf.begin(); it != dialectsOf.end(); ++it) {
      func::FuncOp funcOp = it->first;
      const DialectsSet& dialects { it->second };
      loadDialectsAsAttrs(funcOp, dialects);
    }
  }

private:
  std::map<func::FuncOp, DialectsSet> dialectsOf;

  void addDialect(std::optional<func::FuncOp> op, DialectName dialectName) {
    if (!op) return;
    dialectsOf[*op].insert(dialectName);
  }

  void addDialects(std::optional<func::FuncOp> op, const DialectsSet& dialects) {
    if (!op) return;

    DialectsSet& opDialects { dialectsOf[*op] };
    for (const auto& d : dialects)
      opDialects.insert(d);
  }

  void loadDialectsAsAttrs(func::FuncOp op, const DialectsSet& dialects) {
    // create dictionary entry if not present
    assert(!op->hasAttr(DIALECT_ATTR));

    // construct the dialects dictionary
    std::vector<Attribute> dialectsAttrs;
    for (const DialectName& d : dialects)
      dialectsAttrs.push_back(getStringAttr(op, d));

    // set the attributes of the operation
    op->setDiscardableAttr(
      getStringAttr(op, DIALECT_ATTR), 
      ArrayAttr::get(op->getContext(), llvm::ArrayRef<Attribute>(dialectsAttrs))
    );
  }

  static StringAttr getStringAttr(Operation *op, const std::string& s) {
    return StringAttr::get(op->getContext(), llvm::Twine(s));
  }
};
} // namespace

namespace mlir {
void registerComputeDialectsPass() {
  PassRegistration<ComputeDialectsPass>();
}
} // namespace mlir