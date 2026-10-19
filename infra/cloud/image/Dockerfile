# syntax=docker/dockerfile:1
# Release image of the WSmart-Route cloud runner (CR-03, #123).
#
# Build context is a release tree, not the repository:
#   infra/cloud/image/build.sh                     (builds the tree and the image, tagged by commit)
# or by hand:
#   python logic/package/build_release.py --build build/release
#   docker build -f infra/cloud/image/Dockerfile -t wsmart-route:<commit> build/release
#
# No data and no model weights are baked in: a task's inputs are staged to object
# storage at submit and fetched by the runtime into /app (see logic/controllers/cloud/store/staging.py).
FROM python:3.12-slim

ARG EXTRAS=solvers,cloud
ARG WSR_COMMIT=unknown
LABEL org.opencontainers.image.title="wsmart-route" \
      org.opencontainers.image.revision="${WSR_COMMIT}" \
      org.opencontainers.image.source="https://github.com/ACFHarbinger/WSmart-Route"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WSR_CLOUD_PROJECT_ROOT=/app \
    WSR_CLOUD_WORKDIR=/work \
    WSR_IMAGE_COMMIT=${WSR_COMMIT}

WORKDIR /app
# Dependencies first, from the release pyproject alone, so a code change reuses this layer.
COPY pyproject.toml /tmp/pyproject.toml
RUN python - "${EXTRAS}" > /tmp/requirements.txt <<'PY'
import sys, tomllib
project = tomllib.load(open("/tmp/pyproject.toml", "rb"))["project"]
extras = [name for name in sys.argv[1].split(",") if name]
print("\n".join(project.get("dependencies", []) + [d for e in extras for d in project["optional-dependencies"][e]]))
PY
RUN pip install -r /tmp/requirements.txt
COPY . /app
RUN pip install --no-deps /app
# The runtime fetches staged inputs into /app and writes outputs under /work. Both are
# owned by group 0 and group-writable, so the image also runs under an arbitrary UID
# (`docker run --user <uid>:0`, OpenShift-style orchestrators).
RUN useradd --create-home --uid 10001 --gid 0 wsr \
 && mkdir -p /work \
 && chown -R wsr:0 /app /work \
 && chmod -R g=u /app /work
USER wsr
ENTRYPOINT ["python", "-m", "logic.controllers.cloud.runtime"]
