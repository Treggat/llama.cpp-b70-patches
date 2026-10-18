# Archived: llama.cpp B70 patches

**This fork is archived and read-only. Use the
[B70 llama.cpp Cookbook](https://github.com/Treggat/b70-llamacpp-cookbook) instead.**

This was our working fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for running a dense 27B hybrid
(Qwen3.8-27B with MTP heads and vision) on one Intel Arc Pro B70. The work is finished. We are not taking
issues or pull requests, and the fork will not follow upstream.

The cookbook has everything you need without this repo: all 27 of our patches as files on top of upstream tag
`b11095`, the build and launch recipes, the environment switches, measured numbers and dead ends. Applying
those patches to `b11095` gives exactly the source tree of this fork's last commit.

## What is left here

- **Branch `build/b11095-seats-cachefix`**, last commit `14964cee31`: the same 27 commits as git history.
- **`master`**: a mirror of upstream as of the archive date, with this notice in place of upstream's README.

The original README is upstream's; read it at [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

MIT, as upstream.
