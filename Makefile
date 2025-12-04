.PHONY: prepare-for-kaggle prepare-for-kaggle-commit upload-to-kaggle

prepare-for-kaggle:
	./scripts/prepare_for_kaggle.sh --zip

prepare-for-kaggle-commit:
	./scripts/prepare_for_kaggle.sh --zip --commit

upload-to-kaggle:
	.venv/bin/python scripts/upload_to_kaggle.py

upload-to-kaggle-custom:
	@echo "Usage: make upload-to-kaggle-custom DATASET=name NOTEBOOK=name"
	@echo "Example: make upload-to-kaggle-custom DATASET=nfl-model-v2 NOTEBOOK=nfl-inference-v2"
	.venv/bin/python scripts/upload_to_kaggle.py --dataset-name $(DATASET) --notebook-name $(NOTEBOOK)

