# KWJ Impact Pack Addon

This is a small ComfyUI custom node addon that depends on
[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) being installed separately.

It does not register itself as Impact Pack and does not load Impact Pack's node table. Install the official
Impact Pack normally, then install this folder as a separate custom node directory.

## Nodes

- `KWJ_SEGSFilterClosestMask` / `SEGS Filter (closest mask)`

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

- The internal node type is prefixed with `KWJ_` to avoid collisions with Impact Pack nodes.
- If workflows were saved with the unprefixed `SEGSFilterClosestMask` type, replace that node with
  `KWJ_SEGSFilterClosestMask` after installing this addon.
- This addon imports `impact.core` from the separately installed Impact Pack. If Impact Pack is missing,
  ComfyUI will report that dependency at startup.
