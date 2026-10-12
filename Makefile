## NIDS Makefile — common development commands
.PHONY: install test preprocess train-binary train-multi shap dashboard clean

install:
	pip install -e . --break-system-packages
	pip install -r requirements.txt --break-system-packages

test:
	pytest tests/ -v --tb=short

preprocess:
	python -m src.data.preprocess --raw-dir data/raw --out-dir data/processed

train-binary:
	python -m src.models.train_binary --data-dir data/processed

train-multi:
	python -m src.models.train_multiclass --data-dir data/processed

train-autoencoder:
	python -m src.anomaly.train_autoencoder --data-dir data/processed

shap:
	python -m src.explainability.run_shap --model random_forest --mode binary

dashboard:
	streamlit run src/dashboard/app.py

# Full pipeline
pipeline: preprocess train-binary train-multi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
