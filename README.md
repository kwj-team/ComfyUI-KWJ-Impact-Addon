# KWJ Impact Pack Addon

This is a small ComfyUI custom node addon that depends on
[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) being installed separately.

It does not register itself as Impact Pack and does not load Impact Pack's node table. Install the official
Impact Pack normally, then install this folder as a separate custom node directory.

## Nodes

- `KWJ_SEGSFilterClosestMask` / `SEGS Filter (closest mask)`
- `SEGSFilterClosestMask` / `SEGS Filter (closest mask, legacy)` when that node ID is not already registered

## Installation

1. Install the official `ComfyUI-Impact-Pack` in `ComfyUI/custom_nodes`.
2. Install this addon in a different folder name, for example:

   ```bash
   cd ComfyUI/custom_nodes
   git clone <this-repo-url> ComfyUI-KWJ-Impact-Addon
   ```

3. Restart ComfyUI.

The node appears under `KWJ/ImpactPack/Operation`.

## Notes

- The primary node type is prefixed with `KWJ_` to avoid collisions with Impact Pack nodes.
- Workflows saved with the unprefixed `SEGSFilterClosestMask` type can keep loading through the legacy alias
  when no other installed node has already registered that ID.
- If the official Impact Pack also registers `SEGSFilterClosestMask`, whichever extension ComfyUI loads for that
  ID will be used. Use `KWJ_SEGSFilterClosestMask` in new workflows when you need this addon's implementation.
- This addon imports `impact.core` from the separately installed Impact Pack. If Impact Pack is missing,
  ComfyUI will report that dependency at startup.
