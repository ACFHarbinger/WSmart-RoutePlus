#!/usr/bin/env bash
# Build the release tree and the cloud image, tagged by commit (CR-03, #123).
#   EXTRAS=solvers,cloud,neural infra/cloud/image/build.sh      # add the Attention Model's stack
# Prints the image id; push it with logic/controllers/cloud/images.py.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
COMMIT=$(git -C "$ROOT" rev-parse --short=12 HEAD)
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    COMMIT="${COMMIT}-dirty"
fi
OUT=${OUT:-"$ROOT/build/release-$COMMIT"}
PYTHON=${PYTHON:-python}
rm -rf "$OUT"
"$PYTHON" "$ROOT/logic/package/build_release.py" --build "$OUT"
docker build -f "$ROOT/infra/cloud/image/Dockerfile" \
    --build-arg "WSR_COMMIT=$COMMIT" --build-arg "EXTRAS=${EXTRAS:-solvers,cloud}" \
    -t "wsmart-route:$COMMIT" "$OUT"
docker image inspect --format '{{.Id}}' "wsmart-route:$COMMIT"
