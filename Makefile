.PHONY: prepare-for-kaggle prepare-for-kaggle-commit

prepare-for-kaggle:
	./scripts/prepare_for_kaggle.sh --zip

prepare-for-kaggle-commit:
	./scripts/prepare_for_kaggle.sh --zip --commit
