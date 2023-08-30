// #include "llvm/IR/InstVisitor.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace {

/// This pass illustrates the IR nesting through printing.
struct HaskellPrintingPass
    : public PassWrapper<HaskellPrintingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaskellPrintingPass)

  StringRef getArgument() const final { return "to-haskell"; }
  StringRef getDescription() const final { return "Print into Haskell embedding."; }

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
	printOpLine(printIndent(), op) << "\n";

    // Recurse into each of the regions attached to the operation.
    for (Region &region : op->getRegions())
      printRegion(region);
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
	auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Block main role is to hold a list of Operations: let's recurse.
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };

  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "    ";
    return llvm::outs();
  }

  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

private:
  llvm::raw_ostream& printOpLine(llvm::raw_ostream &stream, Operation *op) {
	// print the operation, polymorphic on operation type
	llvm::StringRef opName {op->getName().getStringRef()};

	HaskellOpPrinter hp{stream};
	if (hp.isStandard(opName)) hp.printStandard(opName, op);
	else if (hp.supportedCasting(opName)) hp.printCasting(op);
	else if (op->hasTrait<mlir::OpTrait::IsTerminator>()) hp.printTerminator(opName, op);
	else if (op->hasTrait<mlir::OpTrait::ConstantLike>()) hp.printConstantLike(op);
	else if (opName.equals("func.func")) hp.print(llvm::dyn_cast<func::FuncOp>(op));
	else if (opName.equals("func.call")) hp.print(llvm::dyn_cast<func::CallOp>(op));
	else if (opName.equals("memref.alloc")) hp.print(llvm::dyn_cast<memref::AllocOp>(op));
	else stream << "UNIMPLEMENTED " << op->getName();

	return stream;
  }

  struct HaskellOpPrinter {
  public:
	HaskellOpPrinter(llvm::raw_ostream &stream) : pStream{&stream} {}; 

  private:
	llvm::raw_ostream *pStream;

	/* access stream by reference */
	llvm::raw_ostream &stream() { return *pStream; }; 

	/* operations to be printed in the "Standard" form */
	static llvm::StringMap<std::string> standardOps() {
		return {
			{"arith.addf", "add"},
			{"arith.addi", "add"},
			{"arith.mulf", "mult"},
			{"arith.negf", "negative"},
			{"arith.divf", "divide"},
			{"arith.uitofp", "intToFp"},
			{"index.sub", "sub"},
			{"memref.dim", "dim"},
			{"memref.dealloc", "deallocMemref"}
		};
	};

	/* casting ops have trivial embedding in Haskell, and will be translated as `return` */
	static llvm::StringSet<> castingOps() {
		return {
			"index.casts",
			"memref.cast"
		};
	};

	/* terminators ops, they all are printed in similar ways */
	static llvm::StringMap<std::string> terminatorOps() {
		return {
			{"func.return", "return"},
			{"scf.yield", "scfYield"},
			{"affine.yield", "affineYield"}
		};
	};

	/* type printing */
	static std::string embedType(mlir::Type t) {
		if (t.isa<mlir::Float32Type>()) return "Float";
		else if (t.isa<mlir::Float64Type>()) return "Double";
		else if (t.isa<mlir::IndexType>()) return "IxType";
		else if (t.isa<mlir::IntegerType>()) return "Int";
		else return "UNSUPPORTED_TYPE";
	}

	template <typename Container>
	void printValuesInterleave(const std::string &sep, const Container &c) {
		llvm::interleave(c, stream(), [&](mlir::Value arg) { arg.printAsOperand(stream(), OpPrintingFlags()); }, sep.c_str());
	};

	void printOperandsInterleave(const std::string sep, Operation *op) {
		printValuesInterleave(sep, op->getOperands());
	};

	/*
	if single value, print:
		v0 
	else print:
		(v0, v1, ..., vn)
	*/
	template <typename Container>
	void printValueTuple(const Container &c, size_t tupleLength) {
		if (tupleLength != 1) stream() << "(";
		printValuesInterleave(", ", c);
		if (tupleLength != 1)stream() << ")";
	}

	/* results section, for operations that creates SSA values */
	void printResults(Operation *op) {
		unsigned nValues = op->getNumResults();
		if (nValues == 0) return;
		printValueTuple(op->getResults(), nValues);
	};

	/*
	"Constant"-like case: prints the single value attribute as a return.

	Requirements on ConstantLikeOp:
	- Subclass of Operation*
	- Should have a `::getValueAttr()` function.
	*/
	template <typename ConstantLikeOp>
	void printConstantLike_(ConstantLikeOp op) {
		// TODO: maybe also add type printing here
		printResults(op); 
		stream() << " <- return ";
		op.getValueAttr().print(stream(), true);
	}

  public:
	/* Check against predefined list of ops embedded in the "Standard" form. */
	bool isStandard(llvm::StringRef opName) { return standardOps().contains(opName); };

	/* Checks against predefined list of supported casting operations */
	bool supportedCasting(llvm::StringRef opName) { return castingOps().contains(opName); };

	/* supported terminator operations */
	bool supportedTerminator(llvm::StringRef opName) { return terminatorOps().contains(opName); };

	/* 
	"Standard" case: prints into the form:
		{standardOps[opName]} <operand0> <operand1> ... <operandn>
	*/
	void printStandard(llvm::StringRef opName, Operation *op) {
		assert(isStandard(opName) && "printStandard called on operation not within isStandard() list.");
		printResults(op);
		stream() << " <- " << standardOps().at(opName) << " ";
		printOperandsInterleave(" ", op);
	};

	/* 
	"Terminators" case: prints into the form:
		0 or >1 args:
		{terminatorOps[opName]} (<operand0>, <operand1>, ... <operandn>)
		1 arg:
		{terminatorOps[opName]} operand0
	*/
	void printTerminator(llvm::StringRef opName, Operation *op) {
		assert(supportedTerminator(opName) && "printTerminator() called on unsupported operation");
		stream() << terminatorOps().at(opName) << " ";

		unsigned nOperands = op->getNumOperands(); 
		printValueTuple(op->getOperands(), nOperands);
	};

	void printConstantLike(Operation *op) {
		assert(op->hasTrait<mlir::OpTrait::ConstantLike>() && "expected operation with CosntantLike trait");
		llvm::StringRef opName {op->getName().getStringRef()};
		if (opName.equals("index.constant")) printConstantLike_(llvm::dyn_cast<index::ConstantOp>(op));
		else if (opName.equals("arith.constant")) printConstantLike_(llvm::dyn_cast<arith::ConstantOp>(op));
		else stream() << "UNSUPPORTED CONSTANT OPERATION: " << opName;
	};

	void printCasting(Operation *op) {
		const llvm::StringRef opName = op->getName().getStringRef();
		assert(supportedCasting(opName) && "expected a supported casting operation");
		// check for self, ensure Op has supported form to be casted (single input & output)
		assert(op->hasTrait<mlir::CastOpInterface::Trait>() && 
			   op->hasTrait<mlir::OpTrait::OneResult>() && 
			   op->hasTrait<mlir::OpTrait::OneOperand>() &&
			   ("operation '" + opName + "' does not look like a cast. double check?").str().c_str());

		printResults(op);
		stream() << " <- return ";
		printOperandsInterleave("", op); 
	}

	/* Special cases */
	
	void print(memref::AllocOp op) {
		mlir::MemRefType memRefType = op.getType();

		printResults(op); // SSA value
		stream() << " :: MemrefType s " << embedType(memRefType.getElementType()) << " <- allocND ["; 
		llvm::interleave( // print memref sizes
			memRefType.getShape(), 
			stream(), 
			[&](int64_t arg) { stream() << arg; }, 
			", "
		); 
		stream() << "]";
	}

	void print(func::FuncOp op) {
		// TODO: add function signature
		stream() << op.getSymName().str() << " ";

		// print arg names, which are arguments to the first block in the attached region
		Block& firstBlock { op.getRegion().front() };
		printValuesInterleave(" ", firstBlock.getArguments());
		
		// print bit before start of region
		stream() << "= do";
		// region will be printed in the parent call
	};

	void print(func::CallOp op) {
		printResults(op);
		stream() << op.getCallee().str() << " ";
		printOperandsInterleave(" ", op);
	};
  };
};
} // namespace

namespace mlir {
void registerHaskellPrintingPass() {
  PassRegistration<HaskellPrintingPass>();
}
} // namespace mlir