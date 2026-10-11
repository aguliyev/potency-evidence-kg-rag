# Knowledge-graph glossary

- **Source Version:** one immutable supplied document revision, scoped by tenant and project.
- **Active Source Head:** the explicitly declared current complete source version; dates alone never replace it.
- **Assertion:** one atomic, passage-grounded claim with independent chronology and review state.
- **ReviewConclusion:** an explicit reviewer or agency conclusion, not an automatic approval of an extraction.
- **Processing State:** one mutable current status/reason row for a complete scoped passage identity.
- **Jev Response:** immutable seven-question probability response over a redacted passage.
- **Jev Selection:** thresholds and selected bundles derived from a Jev response.
- **Extraction Run:** immutable validated structured-generation result for selected bundles.
- **Current Graph:** the active, scoped projection pointer; structural graph records are not scientific claims.

Versions are content-derived: **embedding** describes model/vector settings; **index** adds exact active passage membership; **Jev response** includes questions and redacted input; **Jev selection** adds thresholds; **extraction** includes model, bundles, prompt/schema, and profile; **graph** includes projection inputs, active heads/runs, omissions, mode, and profile.

Jev thresholds trade recall against cost: lowering a threshold calls extraction more often and may add noise; raising it reduces calls and noise but can miss evidence. Defaults are 0.50 except review conclusions and redaction uncertainty at 0.35.
