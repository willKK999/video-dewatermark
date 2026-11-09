GO ?= go
PYTHON ?= .venv/bin/python
.PHONY: test build infra api worker init-storage smoke integration
build:
	mkdir -p work
	$(GO) build -o work/api ./cmd/api
test:
	$(GO) test ./...
	$(PYTHON) -m pytest worker/tests/test_pipeline.py -q
infra:
	docker compose up -d postgres redis minio
api:
	$(GO) run ./cmd/api
worker:
	$(PYTHON) -m worker.main
init-storage:
	$(PYTHON) -m scripts.init_storage
smoke:
	$(PYTHON) -m scripts.smoke
integration:
	INTEGRATION=1 $(PYTHON) -m pytest worker/tests -q
