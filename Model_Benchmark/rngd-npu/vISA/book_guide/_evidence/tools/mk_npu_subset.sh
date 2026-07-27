#!/usr/bin/env bash
# Build a standalone furiosa-opt kernel package containing only the example
# modules whose #[device] kernels compile for the npu backend, so that
# `cargo furiosa-opt --backend npu test` can actually run them on hardware.
set -eu
SRC=/home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex/furiosa-opt-examples
DST=/home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex_npu

rm -rf "$DST"
mkdir -p "$DST/src" "$DST/tests" "$DST/data/mnist"

# --- source modules ---------------------------------------------------------
MODS_FILE="mnist scatter_gather reshape transpose fetch_commit binary_add param fetch_assertions"
for m in $MODS_FILE; do
  if [ -d "$SRC/src/$m" ]; then cp -a "$SRC/src/$m" "$DST/src/";
  else cp -a "$SRC/src/$m.rs" "$DST/src/"; fi
done

# vector_engine: keep only the `zip` submodule
mkdir -p "$DST/src/vector_engine"
cp -a "$SRC/src/vector_engine/zip.rs" "$DST/src/vector_engine/"
sed -e '/^mod normal;$/d' -e '/^mod reduce;$/d' \
    -e '/^pub use normal::\*;$/d' -e '/^pub use reduce::\*;$/d' \
    "$SRC/src/vector_engine.rs" > "$DST/src/vector_engine.rs"

cat > "$DST/src/lib.rs" <<'RS'
//! NPU-runnable subset of furiosa-opt-examples.
//!
//! Only the example modules whose `#[device]` kernels lower cleanly for the
//! `npu` backend are declared here. `--backend npu` compiles every `#[device]`
//! function in the package ahead of time, so a single un-lowerable kernel
//! anywhere in the crate fails the whole build.

#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod binary_add;
pub mod fetch_assertions;
pub mod fetch_commit;
pub mod mnist;
pub mod param;
pub mod reshape;
pub mod scatter_gather;
pub mod transpose;
pub mod vector_engine;
RS

# --- tests ------------------------------------------------------------------
TESTS="mnist_tests scatter_gather_tests reshape_tests transpose_tests fetch_commit_tests binary_add_tests param_tests fetch_assertions_tests"
for t in $TESTS; do
  sed 's/furiosa_opt_examples/visa_ex_npu/g' "$SRC/tests/$t.rs" > "$DST/tests/$t.rs"
done
cp -a "$SRC/tests/common.rs" "$DST/tests/"
mkdir -p "$DST/tests/vector_engine"
sed 's/furiosa_opt_examples/visa_ex_npu/g' "$SRC/tests/vector_engine/zip.rs" > "$DST/tests/vector_engine/zip.rs"
echo 'mod zip;' > "$DST/tests/vector_engine/mod.rs"
cp -a "$SRC/tests/vector_engine_tests.rs" "$DST/tests/"

# --- data -------------------------------------------------------------------
cp -a "$SRC/data/mnist/mnist.safetensors" "$DST/data/mnist/"

# --- manifest ---------------------------------------------------------------
cat > "$DST/Cargo.toml" <<'TOML'
[package]
name = "visa_ex_npu"
version = "0.1.0"
edition = "2024"
publish = false

# Marks this crate as a furiosa-opt kernel package (required by the 0.4 wrapper).
[package.metadata.furiosa-opt]

[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(backend, values("typecheck", "emulation", "npu"))'] }

[lib]
path = "src/lib.rs"

[dependencies]
furiosa-opt-std = "0.4"

[dev-dependencies]
env_logger = "0.11"
half = { version = "2.3.1", features = ["num-traits"] }
rand = { version = "0.9.0", features = ["small_rng", "alloc"] }
safetensors = "0.4"
tokio = { version = "1.29", features = ["rt", "rt-multi-thread", "sync", "macros"] }
TOML

cp -a /home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex/rust-toolchain.toml "$DST/"

echo "built $DST"
find "$DST" -name "*.rs" | wc -l
