# MVAD Preprocess Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `mvad/` preprocessing pipeline that converts MVAD public train zip archives into group-aware train/val Qwen2.5-Omni JSONL with explicit audios.

**Architecture:** Keep MVAD support outside the existing FakeAVCeleb/MAVOS-DD config runner. Use focused modules for archive unpacking, path parsing, group-aware splitting, audio extraction, JSONL writing, and shell commands.

**Tech Stack:** Python standard library, `ffmpeg`, ms-swift JSONL format, bash.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_mvad_prepare.py`

- [x] Write failing tests for modality parsing, group id normalization, group-aware split, explicit audio JSONL records, and zip unpack manifest.
- [x] Run `python -B tests\test_mvad_prepare.py -v` and verify failure because `mvad` does not exist.

### Task 2: Python Implementation

**Files:**
- Create: `mvad/__init__.py`
- Create: `mvad/common.py`
- Create: `mvad/unzip_archives.py`
- Create: `mvad/build_index_and_split.py`
- Create: `mvad/build_av_jsonl.py`
- Create: `mvad/prepare_mvad.py`

- [x] Implement path parsing, labels, group ids, unzip manifest, split outputs, audio extraction, and integrated CLI.
- [x] Run `python -B tests\test_mvad_prepare.py -v` and verify pass.

### Task 3: Commands And Docs

**Files:**
- Create: `mvad/README.md`
- Create: `mvad/run_prepare_mvad.sh`
- Create: `mvad/train_stage1_MVAD.sh`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/commands.md`
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-05.md`

- [x] Add one-command preprocessing and stage1 training scripts.
- [x] Document the train-only internal validation baseline limitation.

### Task 4: Verification

**Files:**
- No code changes expected.

- [x] Run MVAD tests.
- [x] Run py_compile for MVAD modules.
- [x] Run integrated CLI dry-run against temporary zip fixtures.
- [x] Check git status.
