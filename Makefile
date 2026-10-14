# Day-to-day targets. Remote targets need config/spark.env (make config).
-include config/spark.env
SPARK := $(SPARK_USER)@$(SPARK_HOST)
REMOTE := $(or $(REMOTE_DIR),clone)
BACKEND ?= vllm
MODEL ?= gpt-oss-120b
ARGS ?=

.PHONY: help config deploy bootstrap ssh status start stop restart logs switch vllm-switch pull bench tunnel panel smoke agent-install agent clone-command doctor test

help: ## this list
	@grep -hE '^[a-z -]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

config: ## create config/spark.env from the example with a fresh API key
	@if [ -f config/spark.env ]; then echo "config/spark.env exists"; else \
	  cp config/spark.env.example config/spark.env && \
	  sed -i '' "s/^API_KEY=.*/API_KEY=$$(python3 -c 'import secrets; print(secrets.token_hex(16))')/" config/spark.env && \
	  echo "wrote config/spark.env; edit host, user, backend, model"; fi

deploy: ## rsync this tree to the Spark
	scripts/deploy.sh

bootstrap: ## rsync, then run spark/bootstrap.sh there (ARGS="--no-pull" etc.)
	scripts/deploy.sh --bootstrap $(ARGS)

ssh: ## shell on the Spark, in the project tree
	ssh -t $(SPARK) 'cd ~/$(REMOTE) && exec $$SHELL -l'

status: ## memory, GPU, services, endpoints
	scripts/status.sh

start stop restart: ## services for BACKEND (and the sidekick if configured)
	ssh $(SPARK) 'cd ~/$(REMOTE) && if [ "$(BACKEND)" = vllm ]; then systemctl --user $@ vllm.service; else spark/service.sh $@ main; fi; if [ -n "$(SIDEKICK_MODEL)" ]; then spark/service.sh $@ sidekick; fi'

logs: ## follow the main server log
	ssh -t $(SPARK) 'cd ~/$(REMOTE) && if [ "$(BACKEND)" = vllm ]; then journalctl --user -u vllm -f -n 100; else spark/service.sh logs main; fi'

switch: ## MODEL=key  switch the llama.cpp profile and restart (BACKEND=llama)
	ssh -t $(SPARK) 'cd ~/$(REMOTE) && spark/service.sh switch $(MODEL)'

vllm-switch: ## PROFILE=key [LONG=0|1|tokens]  switch the vLLM profile and restart (here and on the Spark)
	@test -n "$(PROFILE)" || { echo "usage: make vllm-switch PROFILE=qwen3.8-27b [LONG=950000]"; exit 1; }
	sed -i '' -e 's/^BACKEND=.*/BACKEND=vllm/' -e 's/^VLLM_PROFILE=.*/VLLM_PROFILE=$(PROFILE)/' -e 's/^VLLM_LONG=.*/VLLM_LONG=$(strip $(or $(LONG),$(VLLM_LONG),0))/' config/spark.env
	ssh -t $(SPARK) 'cd ~/$(REMOTE) && spark/vllm.sh switch $(PROFILE) $(strip $(or $(LONG),$(VLLM_LONG),0))'

pull: ## MODEL=key  download a GGUF profile
	ssh -t $(SPARK) 'cd ~/$(REMOTE) && spark/pull-model.sh $(MODEL)'

bench: ## MODEL=key  llama-bench numbers (stop the server first)
	ssh -t $(SPARK) 'cd ~/$(REMOTE) && spark/bench.sh $(MODEL)'

tunnel: ## forward the model ports to localhost (leave running)
	scripts/tunnel.sh

panel: ## open the control page on the Spark
	open http://$(strip $(SPARK_HOST)):$(or $(strip $(PANEL_PORT)),8088)

smoke: ## tool-calling smoke test through the tunnel
	python3 scripts/smoke-test.py --base-url http://localhost:$(PORT)/v1 --api-key $(API_KEY)

agent-install: ## create the agent's venv
	cd agent && uv sync

agent: ## run the agent (through the tunnel, or set FABLE_BASE_URL)
	cd agent && uv run fable

clone-command: ## link bin/clone into ~/.local/bin: then `clone` in any directory starts the agent there
	mkdir -p $(HOME)/.local/bin && ln -sfn $(CURDIR)/bin/clone $(HOME)/.local/bin/clone
	@echo "clone is in ~/.local/bin; cd into a project and run clone"

doctor: ## check the server from the agent's point of view
	cd agent && uv run fable --doctor

test: ## agent unit tests
	cd agent && uv run pytest -q
