#!/usr/bin/env bash

# MiniTensor installer
#
# Cross-platform (Linux, macOS, Windows via Git Bash/WSL) install script.
# It will:
# - Ensure Python 3.10+, pip, and optionally create a virtual environment (.venv by default)
# - Ensure Rust toolchain (rustup/cargo)
# - Install maturin (with patchelf on Linux)
# - Build and install MiniTensor into the selected Python environment (release by default)
#
# Usage:
#   bash install.sh [--no-venv | --venv <path>] [--debug | --release] [--python <path>]
#
# Notes for Windows:
# - Run this script from Git Bash or WSL. From PowerShell, you can call: bash install.sh
# - Ensure you have a C toolchain (Visual Studio Build Tools) if building native deps is required.
#
# Examples:
#   bash install.sh                  # Create .venv and install release build
#   bash install.sh --debug          # Create .venv and install debug build
#   bash install.sh --no-venv        # Use current Python environment
#   bash install.sh --venv .myvenv   # Use/create a specific venv path
#   bash install.sh --python /usr/bin/python3.12  # Use a specific Python

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Defaults
USE_VENV=1
VENV_DIR=".venv"
RELEASE_BUILD=1
USER_PYTHON=""

color() { # $1=color, $2=message
  local c="$1"; shift
  case "$c" in
    green) printf "\033[0;32m%s\033[0m\n" "$*" ;;
    yellow) printf "\033[0;33m%s\033[0m\n" "$*" ;;
    red) printf "\033[0;31m%s\033[0m\n" "$*" ;;
    blue) printf "\033[0;34m%s\033[0m\n" "$*" ;;
    *) printf "%s\n" "$*" ;;
  esac
}

die() { color red "Error: $*"; exit 1; }

usage() {
  cat <<EOF
MiniTensor installer

Usage: bash install.sh [options]

Options:
  --no-venv            Install into the current Python environment
  --venv <path>        Create/use a virtual environment at <path> (default: .venv)
  --python <path>      Use a specific Python executable for creating venv
  --release            Build in release mode (default)
  --debug              Build in debug mode
  -h, --help           Show this help and exit

Examples:
  bash install.sh
  bash install.sh --venv .myvenv
  bash install.sh --no-venv --debug
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv) USE_VENV=0; shift ;;
    --venv) VENV_DIR="${2:-}"; [[ -z "$VENV_DIR" ]] && die "--venv requires a path"; shift 2 ;;
    --python) USER_PYTHON="${2:-}"; [[ -z "$USER_PYTHON" ]] && die "--python requires a path"; shift 2 ;;
    --release) RELEASE_BUILD=1; shift ;;
    --debug) RELEASE_BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# Detect platform
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
IS_LINUX=0; IS_DARWIN=0; IS_WINDOWS=0; IS_WSL=0
case "$UNAME_S" in
  Linux*) IS_LINUX=1 ;;
  Darwin*) IS_DARWIN=1 ;;
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
esac
if [[ $IS_LINUX -eq 1 ]] && grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=1
fi

color blue "Installing MiniTensor (platform: $UNAME_S)"

# Choose a Python to bootstrap
pick_python_cmd() {
  if [[ -n "$USER_PYTHON" ]]; then
    echo "$USER_PYTHON"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  elif [[ $IS_WINDOWS -eq 1 ]] && command -v py >/dev/null 2>&1; then
    echo "py -3"
  else
    die "Python 3.10+ not found. Please install Python and re-run."
  fi
}

BASE_PYTHON="$(pick_python_cmd)"

# Verify Python version >= 3.10
check_py_ver() {
  local cmd="$1"
  local v
  v="$($cmd - <<'PY'
import sys
print("%d.%d" % (sys.version_info[0], sys.version_info[1]))
PY
  )" || die "Failed to run Python"
  local major="${v%%.*}"; local minor="${v##*.}"
  if (( major < 3 || (major == 3 && minor < 10) )); then
    die "Python >= 3.10 required (found ${v})"
  fi
}

check_py_ver "$BASE_PYTHON"

# Create or select environment
VENV_PY="$BASE_PYTHON"  # default to system if --no-venv
VENV_BIN=""
if [[ $USE_VENV -eq 1 ]]; then
  color blue "Setting up virtual environment at: $VENV_DIR"
  "$BASE_PYTHON" -m venv "$VENV_DIR" || die "Failed to create venv"
  # Compute venv python/bin paths
  if [[ $IS_WINDOWS -eq 1 ]]; then
    VENV_BIN="$VENV_DIR/Scripts"
    VENV_PY="$VENV_BIN/python.exe"
  else
    VENV_BIN="$VENV_DIR/bin"
    VENV_PY="$VENV_BIN/python"
  fi
fi

color blue "Using Python: $VENV_PY"

# Upgrade pip and install maturin
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

MATURIN_EXTRA=""
if [[ $IS_LINUX -eq 1 && $IS_WSL -eq 0 ]]; then
  MATURIN_EXTRA="[patchelf]"
fi

"$VENV_PY" -m pip install "maturin${MATURIN_EXTRA}"

# Ensure Rust toolchain
ensure_rust() {
  if command -v cargo >/dev/null 2>&1; then
    color green "Rust/Cargo found"
    return 0
  fi
  color yellow "Rust not found. Installing via rustup..."
  if ! command -v curl >/dev/null 2>&1; then
    die "curl is required to install rustup. Please install curl and re-run."
  fi
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || die "rustup installation failed"
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env" || true
  if ! command -v cargo >/dev/null 2>&1; then
    die "Cargo not available after rustup installation. Please restart your shell and re-run."
  fi
}

ensure_rust

# Build and install
BUILD_ARGS=(develop)
if [[ $RELEASE_BUILD -eq 1 ]]; then
  BUILD_ARGS+=(--release)
fi

color blue "Building and installing with maturin: ${BUILD_ARGS[*]}"
"$VENV_BIN/maturin" "${BUILD_ARGS[@]}" || "$VENV_PY" -m maturin "${BUILD_ARGS[@]}"

# Validate installation
color blue "Verifying installation..."
"$VENV_PY" - <<'PY'
import sys
try:
    import minitensor as mt
    v = getattr(mt, "__version__", "unknown")
    print(f"MiniTensor installed successfully. Version: {v}")
except Exception as e:
    print("Failed to import minitensor:", e)
    sys.exit(1)
PY

color green "Done."

if [[ $USE_VENV -eq 1 ]]; then
  if [[ $IS_WINDOWS -eq 1 ]]; then
    color yellow "To use this environment next time, activate it with:"
    echo "  source \"$VENV_DIR/Scripts/activate\"    # in Git Bash"
    echo "  . \"$VENV_DIR/Scripts/activate\"          # alternative"
  else
    color yellow "To use this environment next time, activate it with:"
    echo "  source \"$VENV_DIR/bin/activate\""
  fi
fi

if [[ $IS_WINDOWS -eq 1 ]]; then
  color yellow "Note: On Windows, run this script from Git Bash or WSL. From PowerShell: bash install.sh"
fi
