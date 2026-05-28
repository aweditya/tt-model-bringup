# tt-model-bringup — common commands.
# Host targets run on the Tenstorrent box ($(TT_HOST), default qb1) over ssh;
# the device env block + mesh reset live in scripts/run_remote.sh.
TT_HOST ?= qb1
PY      ?= experiments/cb_validate_27b.py   # script for `make run` / `make dr`

.PHONY: help setup lint fmt deploy run dr reset
help:  ## list targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## / — /' | sort

setup:  ## install Python deps into .venv via uv (does NOT build ttnn — see README)
	uv sync

lint:  ## ruff lint (host-independent; never runs device code)
	uvx ruff check .

fmt:  ## ruff auto-format
	uvx ruff format .

deploy:  ## rsync code to the TT host: make deploy [P="path ..."]
	@TT_HOST=$(TT_HOST) scripts/deploy.sh $(P)

run:  ## run a script on the TT host: make run PY=experiments/foo.py
	@TT_HOST=$(TT_HOST) scripts/run_remote.sh $(PY)

dr:  ## deploy default code then run PY on the TT host (edit -> dr loop)
	@TT_HOST=$(TT_HOST) scripts/deploy.sh && TT_HOST=$(TT_HOST) scripts/run_remote.sh $(PY)

reset:  ## reset the TT mesh (recover wedged fabric)
	ssh $(TT_HOST) 'tt-smi -r 0,1,2,3'
