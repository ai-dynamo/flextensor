<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Auto-appended to every page by pymdownx.snippets for tooltip definitions. -->
<!-- Keep entries sorted alphabetically. See docs/api/glossary.md for full descriptions. -->

*[assignment strategy]: Algorithm that maps weights to memory blocks for pipelined transfer (e.g., StrictRoundRobinAssignment).
*[auto trap]: Synonym for forward patching — FlexTensor automatically wraps matched modules' forward methods.
*[direct mode]: Trap implementation that routes parameter access through materialized tensors instead of intercepting every PyTorch operation at dispatch time. Lower overhead than indirect mode.
*[forward patching]: The mechanism where offload() replaces a module's forward method with a wrapper that manages weight transfers. Also called "auto trap."
*[gap trap]: A trap whose module contains no offloadable weights. Gap traps extend the transfer window for neighboring traps.
*[indirect mode]: Trap implementation that intercepts PyTorch operations via __torch_function__ to replace CPU tensors with GPU copies on the fly. More flexible than direct mode but higher overhead.
*[inference phase]: Third active phase — applies the computed offloading strategy for production execution. No timing collection.
*[inner tensor field]: A tensor attached as an attribute to another tensor (e.g., weight.scale for FP8 dequantization). Must be discovered and offloaded together with its parent.
*[transfer_budget_scale]: Multiplier on the time budget available for weight transfers (< 1 adds safety margin, > 1 allows more). Directly controls the budget in latency mode; used as the initial scale in memory mode.
*[latency mode]: Strategy mode activated by max_gpu_mem_fraction=None — optimises for minimum offloading latency with no explicit memory cap.
*[manual trap]: An offload_block() context manager that the user places explicitly around model code, as opposed to auto trap / forward patching.
*[memory block]: A pre-allocated GPU memory region used by block-based transfer modes for pipelined CPU-to-GPU copies.
*[memory mode]: Strategy mode activated by setting max_gpu_mem_fraction to a float — keeps peak GPU usage within a budget.
*[include pattern]: A glob-style string in OffloadConfig.include_patterns that selects which modules or parameters to include for offloading.
*[module execution]: The forward pass of the nn.Module wrapped by a trap — excludes the weight loading and release managed by the trap itself.
*[offload profile]: Serialized result of discovery and profiling (parameter maps, timing, strategy) that can be saved and reloaded to skip those phases.
*[offloading strategy]: Algorithm that decides which weights to keep on GPU vs. move to CPU (e.g., KnapsackStrategy, GreedyStrategy, AdaptiveStrategy).
*[profiling phase]: Second active phase — measures per-trap execution timing using CUDA events across profiling_iters iterations. "Profiling" is the process; the offload profile is the artifact.
*[release strategy]: Algorithm that decides when to free GPU memory after a trap finishes executing its module.
*[source weights]: Original CPU weight copies retained before the first inference loader build so that a strategy replan can rebuild from them. Consumed by a successful rebuild.
*[strategy replan]: Rebuilding the offloading strategy after INFERENCE from timings measured under torch.compile or CUDA graphs, instead of the eager timings collected during profiling.
*[tensor]: In FlexTensor context, a PyTorch tensor (usually a model parameter or buffer) that can be offloaded between GPU and CPU memory.
*[transfer mode]: The mechanism for physically moving weights between CPU and GPU (strategy, allocation_block_transfer, or raw_block_transfer).
*[transfer window]: The time available to pre-fetch the next trap's weights while the current trap's module executes on GPU.
*[trap]: A context manager that wraps a module's forward pass to manage weight loading, timing, and release for that module.
*[discovery phase]: First active phase — discovers which parameters belong to which traps through module ownership, direct getter access, or PyTorch operation interception.
