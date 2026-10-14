# Review-Quality Pipeline Improvements

## Context and Confirmed Gaps

The current graph already performs a limited claim-support check in `verify_citations`, but it sends the verifier extracted research claims and excerpts, not the draft sentences and the citations attached to those sentences. The draft post-check only strips unknown citation keys. `synthesize.py` creates an additional synthesis section; it does not edit, deduplicate, or reconcile the leaf drafts. Research normalizes sources into bibliographic records, but does not retain a source's access mode, provenance tier, or full-text evidence beyond excerpts embedded in claims. Leaf generations run concurrently and receive only local evidence plus dependency summaries.

The agreed direction is:

- Cover all four weaknesses in one integrated implementation.
- Keep parallel leaf drafting for throughput.
- Add a document-wide editorial contract and a whole-document editorial/rewrite pass.
- Treat unsupported citation claims as a blocking quality issue after bounded revision.
- Enforce a tiered scholarly source policy: prioritize peer-reviewed work, canonical books/classics, and authoritative reference works; permit preprints and Wikipedia only with explicit provenance and role labels.

## Ordered Implementation Tasks

1. **Add document-level quality artifacts and schemas.**
   - Extend the outline/planning output with a document thesis or argument arc, terminology/glossary entries, canonical definitions, section handoff goals, and explicit source-policy requirements.
   - Add schemas for a document contract, glossary entry, claim-to-citation support result, source provenance/access metadata, and editorial findings.
   - Preserve backward compatibility for existing saved artifacts by defaulting missing fields and validating old workspaces.
   - Add state/artifact persistence for the contract and quality reports so a run is auditable.

2. **Build the shared coherence contract before research/drafting.**
   - Generate the contract after outline approval and before research fan-out, using the full outline, scope, audience, and section purposes.
   - Include one-sentence thesis, intended progression between sections, controlled terminology with aliases/forbidden variants, concepts that must be defined once, repetition budget, and expected contribution of each leaf.
   - Pass a compact contract view to every research extractor and leaf drafter, alongside local evidence and dependency summaries.
   - Make leaf prompts explicitly distinguish: introduce, develop, compare, or apply a concept; do not redefine terms already owned by another section; use the canonical glossary terms.
   - Add deterministic prompt-size limits and a compact contract representation so the whole outline does not consume the drafting budget.

3. **Add a real editorial pass rather than relying on synthesis.**
   - Introduce a post-draft document editor node after citation verification/critique and before synthesis/assembly.
   - Give the editor the complete outline, contract, all section drafts, evidence summaries, and verification results.
   - Require structured edits/findings covering duplicate definitions, terminology drift, missing transitions, broken argument arc, contradictions, section ordering, unsupported cross-section claims, and uneven depth.
   - Have the editor return revised section markdown (or an explicit no-change result), not only prose for a new “Synthesis” section.
   - Apply edits through a bounded pass; rerun mechanical citation checks and claim-level support checks on changed sections. Do not allow the editor to introduce citation keys absent from the source registry.
   - Keep the existing synthesis node, but redefine it as an optional cross-cutting conclusion generated from the edited drafts and contract. It must not be the sole coherence mechanism.

4. **Implement draft-sentence citation verification.**
   - Parse each draft into factual claim units/sentences with their inline citation groups, retaining section, subsection, and source-key locations for diagnostics.
   - For every cited sentence, provide the verifier the sentence, citation metadata, and the relevant evidence excerpt(s), including source provenance and locator. Ask for entailment/support, scope match, citation completeness, and overclaim severity.
   - Extend verification output beyond `unsupported_claims` to include claim id, cited keys, support judgment, rationale, and severity (`supported`, `partial`, `unsupported`, `unverifiable`).
   - Keep the current mechanical unresolved-key check, but make unsupported/partial factual claims major or blocking according to explicit policy. The selected policy is to block assembly after bounded retries, with unresolved failures retained in the run report.
   - Ensure claims with no evidence, uncited factual assertions, and citations whose excerpt supports only part of a compound sentence are detected. Permit clearly marked interpretation, synthesis, and limitations only under explicit prompt rules.
   - Route sentence-level failures back to the affected section editor/reviser with precise locations and required changes; avoid asking a leaf to repair a document-wide issue.

5. **Strengthen research extraction and source provenance.**
   - Add source fields for provenance tier, source role, peer-review/publication status, access mode, full-text availability, abstract-only status, retrieval URL, and bibliographic confidence.
   - Preserve source excerpts and locators as first-class evidence records, not only free-text observations. Record whether an excerpt came from abstract, full text, book metadata, encyclopedia text, or a web page.
   - Add normalization/adapters for DOI/Crossref/OpenAlex-style metadata and book/reference records; enable OpenAlex or an equivalent scholarly graph for discovery and citation chaining where available.
   - Use source-domain policy in `ResearchBrief` and research prompts: every section should seek a primary/canonical source and an independent corroborating source where the topic supports it.
   - Define the tier policy in configuration: Tier 1 peer-reviewed primary/review articles and canonical books/classics; Tier 2 reputable preprints/working papers and scholarly repositories; Tier 3 authoritative reference works; Tier 4 orientation-only web/Wikipedia. Prevent Tier 4 from satisfying evidence requirements for substantive claims unless explicitly allowed.
   - Add a full-text preference path: fetch article full text or structured sections when available, then fall back to abstract/metadata with an `abstract_only` flag and lower confidence. The research agent must not cite a source as supporting a detailed claim when only metadata or an unrelated abstract was retrieved.
   - Make paywalled/inaccessible records usable for bibliographic discovery but not claim-level support unless an adequate excerpt is available; expose this distinction in prompts and reports.

6. **Revise synthesis and assembly behavior.**
   - Feed the edited drafts, contract, glossary, source-quality report, and citation verification report to the synthesizer.
   - Require synthesis to reference existing supported claims, identify genuine tensions, and avoid restating section introductions. Validate its citations with the same sentence-level verifier before assembly.
   - Make assembly fail or produce a clearly marked incomplete artifact when blocking citation/editorial issues remain; do not silently strip unsupported citations and continue.
   - Include contract violations, source-tier coverage, abstract-only citations, unsupported claims, and unresolved issues in `run_report.json` and the manifest.

7. **Add tests and observability before enabling the stricter gate.**
   - Unit-test contract generation/defaulting, glossary propagation, terminology normalization, and prompt context limits.
   - Unit-test sentence/citation parsing for single, grouped, multiline, and compound citations, plus uncited factual sentences.
   - Add verifier tests for supported, partial, unsupported, abstract-only, and inaccessible-source cases; assert blocking routing and precise issue locations.
   - Add tests proving the editor sees sibling leaves and can remove duplicate definitions while preserving headings and citations.
   - Add tests proving synthesis receives edited drafts and cannot repair coherence merely by appending a second overview.
   - Add source-policy tests for tier classification, Wikipedia orientation-only behavior, full-text versus abstract provenance, book/classic metadata, deduplication, and bibliographic-only records.
   - Extend mocked integration runs to assert the graph order: contract → parallel research/draft → verification/editorial pass → re-verification → synthesis → assembly, and that assembly is withheld for unresolved blocking issues.
   - Log per-section citation support rates, unsupported/partial counts, source tiers, full-text/abstract-only ratios, terminology violations, duplicate-definition findings, editor changes, and final blocking status.

## Affected Boundaries

- `src/scholastra/models/schemas.py` and `src/scholastra/graph/state.py`: new contract, provenance, claim-support, editorial, and report structures.
- `src/scholastra/graph/nodes/outline.py`, `research.py`, `draft.py`, and `subgraphs/*`: contract creation/propagation and source-policy enforcement.
- `src/scholastra/graph/nodes/quality.py`: sentence-level verifier, blocking route, and re-verification.
- New editorial node/prompt plus `graph/build.py`: whole-document editing stage and changed-section routing.
- `src/scholastra/graph/nodes/synthesize.py` and `assembly.py`: consume edited drafts and enforce quality gate.
- `src/scholastra/sources/registry.py`, MCP configuration, and research prompts: provenance/full-text/tier handling.
- `config/presets/*.yaml`: source-policy thresholds, editor/reverification settings, and strict gate defaults.
- `tests/unit/*` and `tests/integration/*`: schema, parser, source-policy, graph-order, and blocking-gate coverage.

## Failure Modes and Rollout

- If contract/editor/verifier LLM calls fail, retain the existing deterministic/mechanical checks, record a blocking quality warning, and do not claim scholarly verification succeeded.
- If full text is unavailable, continue discovery with explicit abstract-only/bibliographic status, but prevent that record from satisfying detailed claim support.
- If the editor exceeds context or budget, return an unchanged draft plus a blocking editorial issue rather than silently assembling incoherent text.
- Roll out behind a preset/config flag first, run mocked and representative real-topic comparisons, then make strict citation gating the default for `mini-book` and `article-long`; keep a clearly labeled audit mode only for exploratory runs.

## Validation Criteria

- A document with sibling leaves using different terms is normalized to glossary terms and receives transitions that reflect the declared argument arc.
- A citation that exists in the registry but does not entail its sentence is reported at sentence/claim location, routed for revision, and blocks assembly if still unresolved.
- A Wikipedia page or abstract-only preprint cannot silently serve as the sole support for a detailed scholarly claim.
- A full-text arXiv/article excerpt is preferred and recorded as such; canonical book/classic metadata is discoverable and labeled distinctly from evidence-bearing excerpts.
- Synthesis changes and reconciles the draft set, while the final synthesis section remains an optional cross-cutting contribution rather than a substitute for editing.
- `run_report.json` makes source quality, citation support, coherence findings, retries, and any remaining blockers inspectable by a human reviewer.
