FROM docker.io/radixark/miles@sha256:cb2a5f35a7de0915cb05fbdb0d2e7a7be0b495eb35c71ef3f10192401bcf18b0

# Keep the base image's compiled SGLang/Megatron/CUDA dependencies intact.
# This source-only override remains a candidate until runtime validation.
RUN cd /root/miles && \
    (git cat-file -e 2233d795dcf167be489d1859971c9233393f04c3^{commit} || \
      git -c http.proxy=http://sys-proxy-rd-relay.byted.org:8118 \
          -c http.lowSpeedLimit=100 -c http.lowSpeedTime=60 \
          fetch --depth 1 origin 2233d795dcf167be489d1859971c9233393f04c3) && \
    git checkout --detach 2233d795dcf167be489d1859971c9233393f04c3 && \
    python3 -m pip install -e . --no-deps --no-build-isolation && \
    python3 -m pip freeze > /root/miles-environment-freeze.txt && \
    python3 -c "import importlib.metadata as m, json; print(json.dumps({n: m.version(n) for n in ('miles', 'ray', 'sglang', 'torch', 'transformer-engine')}, sort_keys=True))" && \
    git rev-parse HEAD > /root/miles-source-commit.txt && \
    git -C /root/Megatron-LM rev-parse HEAD > /root/megatron-source-commit.txt && \
    git -C /sgl-workspace/sglang rev-parse HEAD > /root/sglang-source-commit.txt

# The CPU image builder has no injected libcuda.so.1. Megatron/TE imports are
# validated on the GPU worker by environment_probe.py, not during this build.

LABEL lab.miles.commit="2233d795dcf167be489d1859971c9233393f04c3"
LABEL lab.validation="candidate-environment-not-gpu-validated"
