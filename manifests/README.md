# Evaluation manifest schema

Evaluation manifests are JSON files named `<dataset-alias>.json`. They lock the
dataset identity, split, selected example IDs, and local audio materialization.
The code verifies IDs before decoding.

`example.json` illustrates the schema without redistributing benchmark audio
or text. Replace `audio_path` with a local or relative path produced from the
official dataset. Keep calibration and evaluation IDs disjoint.

