# arraydiff

`arraydiff` is an automatic differentiation library for array-valued operators.
The current implementation consists only of an in-memory representation of the
operator node graph; graph compilation is a non-goal, although graph transforms
(e.g. peephole optimizations) may be possible.

The worldview of `arraydiff` summarized in a few points:
- The only operations that matter are evaluation and various forms of
  differentiation. Generic dataflow is not a goal.
- Sequential evaluation and differentiation must be generically supported.
  Actual recurrent nets arise as a special case.
- There are no explicit graphs of any sort, only references to individual
  operator nodes; the graph is implicit through backtracking links.
- One operator node owns at most one value.
- Operations are idempotent with respect to data transactions.
- Serialized I/O must always be supported.
- Operator nodes ought to be agnostic in theory to data format and backend,
  although in practice there will always be a reference implementation.
