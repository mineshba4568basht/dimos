set -euo pipefail
{
  echo "SHELL: $SHELL"
  echo "PATH: $PATH"
  echo
  echo "python (if any):"; command -v python || true; python --version 2>&1 || true
  echo
  echo "python3:"; command -v python3; python3 --version
  echo
  for v in 3.13 3.12 3.11 3.10 3.9; do
    if command -v python$v >/dev/null 2>&1; then
      echo "python$v: $(command -v python$v) -> $(python$v --version)"
    else
      echo "python$v: NOT FOUND"
    fi
  done
  echo
  echo "pyenv:"; command -v pyenv >/dev/null 2>&1 && pyenv --version || echo "pyenv: NOT FOUND"
  echo "conda:"; command -v conda >/dev/null 2>&1 && conda --version || echo "conda: NOT FOUND"
  echo
  echo "ls /usr/bin/python3* (head):"; ls -l /usr/bin/python3* 2>/dev/null | head -n 40 || true
  echo
  echo "ls /usr/local/bin/python3* (head):"; ls -l /usr/local/bin/python3* 2>/dev/null | head -n 40 || true
} 
