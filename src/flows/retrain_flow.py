# src/flows/retrain_flow.py
"""
Prefect flow that orchestrates the full auto-retraining pipeline.

What is a Prefect flow?
A flow is a Python function decorated with @flow that Prefect tracks,
schedules, and monitors. Inside a flow, each step is a @task —
a tracked unit of work with its own logs, retries, and state.

Pipeline steps:
  1. check_drift_task    -> is new data different from training data?
  2. train_task          -> train a new model on all data up to now
  3. evaluate_task       -> compare new model vs current champion
  4. promote_task        -> promote or reject the new model
"""

from prefect import flow, task, get_run_logger
from src.monitoring.drift_check import check_drift
from src.training.train import train_model
from src.training.evaluate import run_evaluation
from src.registry.promote import run_promotion


# --------------------------------------------------------------------------
# Tasks
# Each @task is one traceable step in the pipeline.
# Prefect logs its start, end, duration, and result automatically.
# --------------------------------------------------------------------------

@task(name="drift-check", retries=1)
def check_drift_task(reference_month: str, new_month: str) -> dict:
    """
    Task wrapper around drift_check.check_drift().

    Why wrap in a task?
    Prefect needs to know about each step separately so it can:
    - Retry on failure (retries=1 means one automatic retry)
    - Show each step's status in the Prefect UI
    - Skip downstream tasks if this one fails
    """
    logger = get_run_logger()
    logger.info(f"Checking drift: {reference_month} -> {new_month}")
    result = check_drift(reference_month, new_month)
    logger.info(f"Drift detected: {result['drift_detected']} | "
                f"Drifted features: {result['n_drifted']}/{result['n_total']}")
    return result


@task(name="train-model", retries=1)
def train_task(up_to_month: str) -> str:
    """
    Task wrapper around train.train_model().
    Returns run_id — passed automatically to the next task.

    This solves the manual run_id problem:
    Prefect passes the return value of this task directly
    to evaluate_task and promote_task as a parameter.
    """
    logger = get_run_logger()
    logger.info(f"Training model on data up to {up_to_month}")
    run_id = train_model(up_to_month=up_to_month)
    logger.info(f"Training complete | run_id: {run_id}")
    return run_id


@task(name="evaluate-model", retries=1)
def evaluate_task(challenger_run_id: str, test_month: str) -> dict:
    """
    Task wrapper around evaluate.run_evaluation().
    Receives run_id automatically from train_task output.
    """
    logger = get_run_logger()
    logger.info(f"Evaluating challenger: {challenger_run_id}")
    result = run_evaluation(
        challenger_run_id=challenger_run_id,
        test_month=test_month,
    )
    logger.info(f"Evaluation decision: {'PROMOTE' if result['promote'] else 'REJECT'}")
    return result


@task(name="promote-model")
def promote_task(challenger_run_id: str, should_promote: bool) -> bool:
    """
    Task wrapper around promote.run_promotion().
    Receives both run_id and promote decision from previous tasks.
    """
    logger = get_run_logger()
    from src.monitoring.metrics import RETRAIN_COUNTER
    promoted = run_promotion(
        challenger_run_id=challenger_run_id,
        should_promote=should_promote,
    )
    result_label = "promoted" if promoted else "rejected"
    RETRAIN_COUNTER.labels(result=result_label).inc()
    logger.info(f"Model {result_label}")
    return promoted


# --------------------------------------------------------------------------
# Main Flow
# --------------------------------------------------------------------------

@flow(name="bike-retrain-pipeline", log_prints=True)
def retrain_pipeline(reference_month: str, new_month: str):
    """
    Full auto-retraining pipeline.

    How data flows between tasks automatically:
        drift_result = check_drift_task(...)
                              |
                              v
                    drift_detected? -> if False: STOP
                              |
                              v (if True)
        run_id = train_task(...)
                              |
                    ----------+----------
                    |                   |
                    v                   v
        eval_result = evaluate_task(run_id, ...)
                    |
                    v
        promote_task(run_id, eval_result["promote"])

    Args:
        reference_month: last month used in training  (e.g. "2011-06")
        new_month      : new incoming month to check  (e.g. "2011-07")
    """
    logger = get_run_logger()
    logger.info(f"Pipeline started | reference: {reference_month} | new: {new_month}")

    # Step 1 — Drift check
    drift_result = check_drift_task(reference_month, new_month)

    if not drift_result["drift_detected"]:
        logger.info("No drift detected — current model is still valid. Pipeline stopped.")
        return {"status": "skipped", "reason": "no_drift"}

    # Step 2 — Train new model
    # run_id flows automatically to the next steps
    run_id = train_task(up_to_month=new_month)

    # Step 3 — Evaluate challenger vs champion
    eval_result = evaluate_task(
        challenger_run_id=run_id,
        test_month=new_month,
    )

    # Step 4 — Promote or reject
    promoted = promote_task(
        challenger_run_id=run_id,
        should_promote=eval_result["promote"],
    )

    status = "promoted" if promoted else "rejected"
    logger.info(f"Pipeline complete | result: {status}")
    return {"status": status, "run_id": run_id}


if __name__ == "__main__":
    # Simulate: model trained on first 6 months, new data = month 7
    result = retrain_pipeline(
        reference_month="2011-06",
        new_month="2011-07",
    )
    print(f"\nPipeline result: {result}")