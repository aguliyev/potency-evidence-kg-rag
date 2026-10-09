# Project-isolation fixture

The two inputs deliberately reuse `source_id`, `source_version` and `passage_id` while changing project scope and marker text. Import both into the same store, restart the process, and demonstrate that retrieval, graph traversal, traces, citations and answers for one project cannot contain the other project's marker.

These are synthetic test records, not part of the FDA evidence corpus.

Import both fixtures into the SAME persistent store without clearing the first. Query each project in vector and graph modes, including after restart. Assert its marker is present and the other marker absent from selected evidence, traversed graph content, citations and answers. The fixtures are separate from the 791 FDA records and seven main answers. Test public-corpus sharing only through explicit design policy; do not allow a fixture query to fall back to another project.

The additional [source-correction and conflict scenarios](BEHAVIOR_CASES.md) have their own test scopes and three separate required answer/trace records. Do not clear one fixture scope merely to hide an isolation failure.
