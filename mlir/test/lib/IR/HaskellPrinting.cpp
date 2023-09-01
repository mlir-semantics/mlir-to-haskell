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

typedef llvm::raw_ostream stream_t; 

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

  stream_t &stream() { return llvm::outs(); };

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };

  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  stream_t &printIndent() {
    for (int i = 0; i < indent; ++i)
      stream() << "    ";
    return stream();
  }

  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

private:
  stream_t& printOpLine(stream_t &stream, Operation *op) {
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
	else if (opName.equals("memref.load")) hp.print(llvm::dyn_cast<memref::LoadOp>(op));
	else if (opName.equals("memref.store")) hp.print(llvm::dyn_cast<memref::StoreOp>(op));
	else stream << "UNIMPLEMENTED " << op->getName();

	return stream;
  }

  struct HaskellOpPrinter {
  public:
	HaskellOpPrinter(stream_t &stream) : pStream{&stream} {}; 

  private:
	stream_t *pStream;

	/* access stream by reference */
	stream_t &stream() { return *pStream; }; 

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
		else if (auto mt = t.dyn_cast<mlir::MemRefType>()) return "MemrefType s " + embedType(mt.getElementType());
		else return "UNSUPPORTED_TYPE";
	}

	void printValue(mlir::Value arg) { arg.printAsOperand(stream(), OpPrintingFlags()); };

	template <typename Container>
	void printValuesInterleave(const std::string &sep, const Container &c) {
		llvm::interleave(c, stream(), [&](mlir::Value arg) { printValue(arg); }, sep.c_str());
	};

	template <typename Container>
	void printValues(const Container &c,
					 const std::string sep = ", ",
					 const std::string openB = "", const std::string closeB = "") {
		stream() << openB;
		printValuesInterleave(sep, c);
		stream() << closeB;
	}

	void printOperands(Operation *op, const std::string sep = " ") { printValues(op->getOperands(), sep); };

	/*
	if single value, print:
		v0 
	else print:
		(v0, v1, ..., vn)
	*/
	template <typename Container>
	void printValueTuple(const Container &c, size_t tupleLength) {
		if (tupleLength == 1) printValues(c, "");
		else printValues(c, ", ", "(", ")");
	};

	template <typename Container>
	void printIndices(const Container &c) { printValues(c, ", ", "[", "]"); };

	/*
	print operation out in the form:
		[(return_vals) <-] <op_name> memref_val[i0, i1, ..., in] 

	MemRefIndexingOp must have the following functions:
		- getMemref()
		- getIndices()
	*/
	template <typename MemRefIndexingOp>
	void printMemRefIndexingOp(MemRefIndexingOp op, const std::string opName) {
		if (printResults(op)) stream() << " <- ";
		stream() << opName << " ";
		printValue(op.getMemref());
		stream() << " ";
		printIndices(op.getIndices());
	}

	/* 
	results section, for operations that creates SSA values 
	returns boolean, if any SSA values were created. 
	*/
	bool printResults(Operation *op) {
		unsigned nValues = op->getNumResults();
		if (nValues == 0) return false;
		printValueTuple(op->getResults(), nValues);
		return true;
	};

	void printResultsWithAssign(Operation *op) { if (printResults(op)) stream() << " <- "; };

	/*
	"Constant"-like case: prints the single value attribute as a return.

	Requirements on ConstantLikeOp:
	- Subclass of Operation*
	- Should have a `::getValueAttr()` function.
	*/
	template <typename ConstantLikeOp>
	void printConstantLike_(ConstantLikeOp op) {
		// TODO: maybe also add type printing here
		printResultsWithAssign(op); 
		stream() << "return ";
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
		printResultsWithAssign(op);
		stream() << standardOps().at(opName) << " ";
		printOperands(op);
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

		printResultsWithAssign(op);
		stream() << "return ";
		printOperands(op); 
	};

	/* Special cases */
	
	void print(memref::AllocOp op) {
		mlir::MemRefType memRefType = op.getType();

		printResults(op); // SSA value
		stream() << " :: " << embedType(memRefType) << " <- allocND [";
		llvm::interleave( // print memref sizes
			memRefType.getShape(), 
			stream(), 
			[&](int64_t arg) { stream() << arg; }, 
			", "
		); 
		stream() << "]";
	};

	void print(memref::LoadOp op) {	printMemRefIndexingOp(op, "load"); };

	void print(memref::StoreOp op) { printMemRefIndexingOp(op, "store"); };

	void print(func::FuncOp op) {
		// TODO: add function signature
		stream() << op.getSymName().str() << " ";

		// print arg names, which are arguments to the first block in the attached region
		Block& firstBlock { op.getRegion().front() };
		printValuesInterleave(" ", firstBlock.getArguments());
		
		// print bit before start of region
		stream() << " = do";
		// region will be printed in the parent call
	};

	void print(func::CallOp op) {
		printResults(op);
		stream() << op.getCallee().str() << " ";
		printOperands(op);
	};
  };
};
} // namespace

namespace mlir {
void registerHaskellPrintingPass() {
  PassRegistration<HaskellPrintingPass>();
}
} // namespace mlir