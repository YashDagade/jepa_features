# Code Quality And Refactoring

Use this guide for new source code and for cleanup of existing modules. The
goal is code that reads like a precise technical explanation while remaining
compact, testable, and faithful to existing behavior.

## Core Standard

1. Start each module with its purpose and its place in the larger workflow.
2. Keep the public path easy to follow: configuration, data models, public
   operations, private helpers, then the command-line entry point.
3. Give functions descriptive names and short docstrings that explain their
   contract. Add comments for decisions, invariants, and non-obvious steps.
4. Do not comment obvious assignments or restate syntax. Tutorial-style
   line-by-line comments belong in documentation examples, not production
   modules.
5. Prefer small pure functions and explicit data models over mutable
   dictionaries, hidden state, or long argument lists.
6. Implement shared behavior once and import it. Delete superseded wrappers,
   duplicate implementations, compatibility aliases, and one-off report
   builders after their callers have migrated.
7. Fail loudly when an invariant is violated. Catch only exceptions that the
   current layer can handle meaningfully.

## Naming And Sections

A name beginning with an underscore, such as `_validate_episode`, is a
**private helper**, not an anonymous function. Python anonymous functions are
`lambda` expressions.

When a module has several private helpers, group them under an explicit
`# Private helpers` section after the public operations they support. A helper
docstring should state:

- what contract or invariant it implements;
- why it is private to this module;
- where the public workflow calls it, when that is not obvious.

Avoid a mechanical private-helper section when it would separate a helper from
the only public function that uses it and make the module harder to read.
Preserve useful workflow section headings and locality. Moving every private
helper to the bottom of a file is not an improvement by itself.

## Configuration Boundary

Experiment variation belongs in strict YAML configuration:

- environment family and room count;
- cue type and marker appearance;
- fixed or sampled geometry;
- expert, search, or random policy;
- action noise, seed, trajectory count, and evaluation ranks.

Python owns algorithms and invariants: scenario sampling, policy execution,
rendering, validation, transactional storage, and publication. Do not encode
named experiments as branches, aliases, or near-duplicate scripts. Config
values must be canonical and explicit; do not accept multiple spellings for
the same mode.

## Refactoring Workflow

1. Record the current public API, configuration contract, schemas, hashes,
   provenance, W&B keys, and tests.
2. Find the underlying repeated operation before moving code.
3. Introduce one shared implementation and migrate callers in a coherent
   change.
4. Delete the replaced implementation in the same milestone.
5. Compare behavior before and after with focused tests and representative
   artifacts.
6. Run the broader affected test suite, then report source lines and file
   counts before and after.

Do not set a line-reduction target that encourages compressed or cryptic code.
Deletion is valuable when it removes duplicated concepts, obsolete paths, or
unnecessary indirection. A refactor is not successful if it merely moves the
same complexity into a denser file.

Reject and revert a pilot that creates broad churn without deleting a concept,
even when its tests pass. If meaningful simplification requires shared
behavior to move across module boundaries, widen the milestone deliberately
instead of compressing one file or adding layers of private helpers.

## Dataset And Storage Contracts

Tunnel Rooms data refactors must preserve:

- HDF5 dataset names, shapes, dtypes, compression, and episode offsets;
- deterministic sampling for the same seed and configuration;
- scenario, collection, and evaluation hashes;
- resumable partial-file and journal behavior;
- source immutability and destination collision checks;
- manifest and provenance fields consumed by training and evaluation;
- action, observation, reward, terminal, noise, and policy semantics.

Dataset variants should share one collection pipeline. Rendering, policies,
scenario generation, validation, and storage should each have one clear owner.

## Generated Artifacts

Temporary analysis code, report builders, generated manifests, videos, plots,
Slurm output, and datasets belong under ignored `runs/`. Commit only durable
source, strict configs, tests, documentation, and small intentional
provenance. Never add a tracked `scripts/reports/` path for one-off reports.

## Acceptance Checklist

- The main workflow can be understood from the public functions alone.
- Public behavior and user-visible W&B output are unchanged unless the task
  explicitly requests a change.
- Each concept has one implementation and one canonical config value.
- Private helpers are named and organized consistently.
- Comments explain intent and invariants, not syntax.
- Focused and affected regression tests pass in the project environment.
- Representative before/after artifacts or hashes agree where exact equality
  is required.
- The change deletes obsolete code and reports net line/file-count impact.
- The coherent, verified milestone is committed and pushed.
