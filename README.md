# SageMaker + MLflow Project Template

What you get
- Train entrypoint (Script Mode).
- MLflow tracking + model registry.
- Batch transform job + real-time endpoint (inference script).
- Drift checks (PSI) and simple canary rollout notes.

Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/synthetic_generator.py
# Local train
python src/train/train.py --config configs/params.yaml
# Register to MLflow
MLFLOW_TRACKING_URI=file:./mlruns python src/train/register.py
# Batch score (local)
python src/inference/batch.py --input data/synth_claims.parquet --out out/preds.parquet

Deploy (notes)
- See `infra/README.md` for CDK/Terraform snippets to create: S3, Role, Training Job, Endpoint.

Disclaimer
Synthetic data only. No employer IP.
