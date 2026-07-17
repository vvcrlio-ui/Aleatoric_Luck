#!/usr/bin/env bash

# Source this file from Bash or Zsh. It locates .venv relative to the project,
# so the project itself can live at a different path on each machine.
if [ -n "${ZSH_VERSION:-}" ]; then
  _ALEATORIC_ACTIVATE_SOURCE="${(%):-%N}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
  _ALEATORIC_ACTIVATE_SOURCE="${BASH_SOURCE[0]}"
else
  echo "activate_env.sh must be sourced from Bash or Zsh." >&2
  return 1 2>/dev/null || exit 1
fi

_ALEATORIC_ROOT="$(cd "$(dirname "$_ALEATORIC_ACTIVATE_SOURCE")" && pwd)"
_ALEATORIC_VENV="${VENV:-$_ALEATORIC_ROOT/.venv}"

if [ ! -x "$_ALEATORIC_VENV/bin/python" ]; then
  echo "Virtual environment not found: $_ALEATORIC_VENV" >&2
  echo "Create it with: ./setup_env.sh" >&2
  unset _ALEATORIC_ACTIVATE_SOURCE _ALEATORIC_ROOT _ALEATORIC_VENV
  return 1 2>/dev/null || exit 1
fi

source "$_ALEATORIC_VENV/bin/activate"
# A venv generated at an older project location may set a stale VIRTUAL_ENV.
# Put the environment found relative to this project back at the front.
export VIRTUAL_ENV="$_ALEATORIC_VENV"
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r 2>/dev/null || true

unset _ALEATORIC_ACTIVATE_SOURCE _ALEATORIC_ROOT _ALEATORIC_VENV
