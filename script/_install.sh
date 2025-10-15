#!/usr/bin/env bash
set -euo pipefail

echo "Installing the necessary packages ..."
pip install -r script/requirements.txt

echo "Installing pytorch3d (from facebookresearch/pytorch3d@stable) ..."
# needs ninja + cmake present in your env; you already have them via conda-forge
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "Patching sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION="$(pip show sapien | awk '/Location/ {print $2}')/sapien"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"

if [[ -f "$URDF_LOADER" ]]; then
  # backup before editing
  cp "$URDF_LOADER" "${URDF_LOADER}.bak"

  # add encoding="utf-8" and fix .srdf extension
  # robust two-step patch (works even if spacing differs)
  sed -i 's/open(urdf_file, "r")/open(urdf_file, "r", encoding="utf-8")/g' "$URDF_LOADER"
  sed -i 's/urdf_file\[:-]4\] \+\"srdf\"/urdf_file[:-4] + ".srdf"/g' "$URDF_LOADER"

  # also ensure srdf open uses utf-8
  sed -i 's/open(srdf_file, "r")/open(srdf_file, "r", encoding="utf-8")/g' "$URDF_LOADER"
else
  echo "WARN: $URDF_LOADER not found; skipping sapien patch."
fi

echo "Patching mplib/planner.py ..."
MPLIB_LOCATION="$(pip show mplib | awk '/Location/ {print $2}')/mplib"
PLANNER="$MPLIB_LOCATION/planner.py"

if [[ -f "$PLANNER" ]]; then
  cp "$PLANNER" "${PLANNER}.bak"
  # remove the 'or collide' part in that specific condition
  sed -i -E 's/(if[[:space:]]+np\.linalg\.norm\(delta_twist\)[[:space:]]*<[^:]*)([[:space:]]*or[[:space:]]*collide)([[:space:]]*or[[:space:]]*not[[:space:]]+within_joint_limit:)/\1\3/' "$PLANNER"
else
  echo "WARN: $PLANNER not found; skipping mplib patch."
fi

echo "Installing cuRobo (skipping if already present) ..."
if python - <<'PY' >/dev/null 2>&1; then
import importlib; import sys
sys.exit(0 if importlib.util.find_spec("curobo") else 1)
PY
then
  echo "cuRobo already importable; skip clone/build."
else
  mkdir -p envs
  if [[ ! -d envs/curobo ]]; then
    git clone https://github.com/NVlabs/curobo.git envs/curobo
  fi
  pushd envs/curobo
  # use your known-good toolchain & CUDA; avoid build isolation
  pip install -e . --no-build-isolation
  popd
fi

echo "Installation of the basic environment complete!"
echo "Next:"
echo "  1) (Important!) Download assets from HuggingFace."
echo "  2) Optionally install extra requirements for baselines."
echo "See INSTALLATION.md for details."
