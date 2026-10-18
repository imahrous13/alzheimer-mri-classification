# Convenience targets for the Alzheimer MRI project.
PYTHON ?= python

.PHONY: help install test compile figures dashboard notebook app pipeline clean

help:
	@echo "Targets:"
	@echo "  install    Install runtime dependencies"
	@echo "  test       Run the unit test suite (includes the Streamlit app smoke test)"
	@echo "  compile    Byte-compile src, scripts, and tests"
	@echo "  pipeline   Reproduce the full pipeline (audit -> split -> models -> notebook -> dashboard)"
	@echo "  figures    Regenerate the docs/figures gallery"
	@echo "  dashboard  Rebuild the HTML dashboard"
	@echo "  notebook   Regenerate and execute the analysis notebook"
	@echo "  app        Launch the Streamlit demo"
	@echo "  clean      Remove Python caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall -q src scripts tests

figures:
	PYTHONPATH=src $(PYTHON) scripts/make_figures.py

dashboard:
	PYTHONPATH=src $(PYTHON) scripts/build_dashboard.py

notebook:
	PYTHONPATH=src $(PYTHON) scripts/build_notebook.py
	PYTHONPATH=src $(PYTHON) -m jupyter nbconvert --to notebook --execute \
		notebooks/alzheimer_mri_audit_and_results.ipynb \
		--output alzheimer_mri_audit_and_results.ipynb \
		--output-dir notebooks --ExecutePreprocessor.timeout=600

app:
	$(PYTHON) -m streamlit run app.py

pipeline:
	bash scripts/run_pipeline.sh

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; print('caches removed')"
