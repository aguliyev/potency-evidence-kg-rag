"""Every LLM prompt text used by the pipeline, in one place.

Pure strings, no imports: ``prompt_resources/*`` assemble them (bundle selection, schema and
relationship tables) and the clients send them. Editing any text here changes the prompt hash
and therefore the extraction version, so affected passages re-run.
"""

from __future__ import annotations

# --- Jev (seven fixed yes/no questions: (question, criterion for "true")) -----------------------
JEV_QUESTIONS: dict[str, tuple[str, str]] = {
    "product_context": ("Does this passage identify product, application, indication, or applicability?", "Identify explicit product context only."),
    "alias_identity": ("Does this passage contain a product name, abbreviation, or alias?", "Capture names without inferring identity."),
    "potency_strategy": ("Does this passage discuss potency strategy?", "Include explicit potency context."),
    "assay_evidence": ("Does this passage describe assays or measured attributes?", "Require an explicit assay or measurement."),
    "regulatory_history": ("Does this passage record a regulatory interaction or chronology?", "Capture proposed, requested, revised, accepted, rejected, or unresolved events."),
    "review_conclusion": ("Does this passage state a reviewer or agency conclusion?", "Require an explicit conclusion or recommendation."),
    "redaction_uncertainty": ("Does this passage contain redaction or unresolved uncertainty?", "Do not reconstruct redacted content."),
}
JEV_FALSE_CRITERION = "The criterion is not explicitly supported."

# --- Structured extraction -----------------------------------------------------------------------
UNIVERSAL_INSTRUCTIONS = """\
You extract structured evidence from one passage of an FDA review document. Extract only what the passage \
explicitly states; never infer, guess, or reconstruct concealed (redacted) content.

Return a single JSON object with four lists: entities, assertions, relationships, source_refs.

Work in this order: (1) entities of the allowed types, (2) assertions, (3) relationships, (4) source_refs. \
Lists may be empty, but do not return empty lists when the passage contains anything of an allowed type: \
an empty answer is correct only when the passage truly has none of them."""

QUOTE_RULES = """\
Quotes (entities, assertions, source_refs):
- supporting_quote must be copied verbatim, character for character, from the passage (keep line breaks, \
spelling, punctuation). It must be one contiguous span: no ellipses, no paraphrase, no joined fragments. \
Keep it short (a phrase or one sentence) so it is easy to match exactly.
- If the exact quote occurs more than once in the passage, set occurrence_index (0-based) to the occurrence \
you mean; otherwise leave occurrence_index null. Prefer a quote that occurs only once.
- Never include a [REDACTED:...] token inside a quote."""

ID_RULES = """\
IDs:
- local_id values are short and unique across ALL records in the response (entities, assertions, \
relationships and source_refs share one namespace), e.g. e1, e2, a1, r1, s1.
- Every reference (target_local_id, source_local_id, assertion_local_id, record_local_id) must name a \
local_id that exists in this same response."""

ASSERTION_RULES = """\
Assertions: one atomic claim per assertion, written as a single sentence on one line (no line breaks), \
with the exact supporting_quote that states it. Split compound statements into separate assertions. \
chronology_state is "reported" unless the passage itself says the event was proposed, requested, revised, \
accepted, rejected or is unresolved."""

# Placeholders: {relation_table}
RELATIONSHIP_RULES = """\
Relationships connect two records. Each one needs: local_id, relation_type, source_local_id, \
target_local_id, assertion_local_id (the Assertion that states the link; required), attributes ({{}} if none).
relation_type MUST be exactly one of the names below and the source and target must have exactly the \
listed types, otherwise the whole response is rejected. Do not invent relation types (for example "supports", \
"describes" or "measured_after_treatment_with" are NOT valid). Use the literal local_id "source_passage" \
for the SourcePassage endpoint. If no listed relation fits, emit no relationship: they are optional.
{relation_table}"""

SOURCE_REF_RULES = """\
source_refs: optional extra quotes for a record. record_local_id must name an existing entity, assertion \
or relationship; supporting_quote follows the quote rules."""

REDACTION_RULES = """\
Redactions: the passage shows concealed text as tokens like [REDACTED:<passage_id>:<n>]. Never guess what \
is behind a token. Do not output RedactionMention records: the system records every token itself."""

RETRY_INSTRUCTION = (
    "The previous response used a type outside the selected bundles. Return a complete "
    "replacement response using only the emphasized allowed types."
)

# Required fields (machine-readable hint appended to the prompt).
TYPE_SCHEMAS: dict[str, dict[str, object]] = {
    "Product": {"required": ("local_id", "name", "supporting_quote")},
    "Application": {"required": ("local_id", "application_id", "supporting_quote")},
    "Indication": {"required": ("local_id", "name", "supporting_quote")},
    "AliasCandidate": {"required": ("local_id", "candidate", "target_local_id", "supporting_quote")},
    "PotencyStrategy": {"required": ("local_id", "description", "supporting_quote")},
    "Assay": {"required": ("local_id", "name", "supporting_quote")},
    "MeasuredAttribute": {"required": ("local_id", "name", "supporting_quote")},
    "RegulatoryInteraction": {"required": ("local_id", "chronology_state", "supporting_quote")},
    "ReviewConclusion": {"required": ("local_id", "conclusion", "supporting_quote")},
    "RedactionMention": {"required": ("local_id",)},  # derived by the system, not emitted by the model
}

# One line per type: what to extract.
TYPE_GUIDANCE: dict[str, str] = {
    "Product": "the medicinal product named in the passage (brand or nonproprietary name); name = the name as written",
    "Application": "a regulatory application identifier such as BLA 125785; application_id = the identifier as written",
    "Indication": "a disease or population the product is indicated for or studied in; name = as written",
    "AliasCandidate": "an alternative name, code or abbreviation of a Product or Assay stated in the passage (e.g. exa-cel, CTX001); candidate = the alias as written; target_local_id = local_id of the Product or Assay entity it names (it must be in this response); do not infer identity that is not stated",
    "PotencyStrategy": "a described approach to demonstrate or measure potency; description = short wording from the passage",
    "Assay": "a named assay or test method; name = as written; role only if the passage states it",
    "MeasuredAttribute": "a quantity or attribute that is measured (e.g. HbF level); name = as written",
    "RegulatoryInteraction": "a meeting, request, agreement, review step or other agency/sponsor exchange; chronology_state = proposed, requested, revised, accepted, rejected, unresolved or reported, as the passage states it; description optional",
    "ReviewConclusion": "an explicit reviewer or agency conclusion or recommendation; conclusion = the conclusion in one sentence",
    "RedactionMention": "(recorded automatically by the system; do not emit)",
}

# --- Optional alias discovery --------------------------------------------------------------------
ALIAS_DISCOVERY_INSTRUCTIONS = (
    "Identify only explicit Product or Assay aliases in the supplied passages. "
    "Return every proposal with the exact passage_id and an exact supporting_quote copied "
    "from that passage. Do not infer aliases, normalize quotes, or approve a proposal."
)
