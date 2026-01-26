"""
Main training script for Google Colab.

This script trains a multi-task BERT-like model for:
1. Detecting Personally Identifiable Information (PII) in text
2. Detecting co-reference clusters (mentions referring to the same entity)

Usage:
    # Basic usage (saves to Google Drive by default)
    python train.py

    # In Colab, or to customize Google Drive settings:
    main(use_google_drive=True, drive_folder="MyDrive/pii_models")

    # To disable Google Drive saving:
    main(use_google_drive=False)
"""

import time

from absl import logging

# Import from local modules
try:
    from .config import EnvironmentSetup, TrainingConfig
    from .preprocessing import DatasetProcessor
    from .trainer import PIITrainer
except ImportError:
    # Fallback for direct execution
    from config import EnvironmentSetup, TrainingConfig
    from preprocessing import DatasetProcessor
    from trainer import PIITrainer


def main(
    use_google_drive: bool = True,
    drive_folder: str = "MyDrive/pii_models",
    training_samples_dir: str | None = None,
):
    """
    Orchestrates environment setup, dataset preparation, model training, evaluation, and optional saving to Google Drive.

    Parameters:
        use_google_drive (bool): If True, attempt to mount Google Drive and save the trained model there (Colab context).
        drive_folder (str): Target folder path in Google Drive where the model will be saved.
        training_samples_dir (str | None): Optional override for the training samples directory; when provided it is passed to TrainingConfig (e.g., "model/dataset/training_samples" or another custom path).
    """
    logging.info("=" * 60)
    logging.info("Multi-Task PII Detection and Co-reference Detection Training")
    logging.info("=" * 60)

    # Setup environment
    logging.info("\n1️⃣  Setting up environment...")

    # Mount Google Drive if requested
    drive_mounted = False
    if use_google_drive:
        drive_mounted = EnvironmentSetup.mount_google_drive()

    EnvironmentSetup.check_gpu()

    # Load configuration
    logging.info("\n2️⃣  Loading configuration...")
    config_kwargs = {}
    if training_samples_dir is not None:
        config_kwargs["training_samples_dir"] = training_samples_dir
    config = TrainingConfig(**config_kwargs)
    config.print_summary()

    # Prepare datasets
    logging.info("\n3️⃣  Preparing datasets...")
    dataset_processor = DatasetProcessor(config)
    train_dataset, val_dataset, mappings, coref_info = (
        dataset_processor.prepare_datasets()
    )

    # Initialize trainer
    logging.info("\n4️⃣  Initializing trainer...")
    trainer = PIITrainer(config)
    trainer.load_label_mappings(mappings, coref_info)
    trainer.initialize_model()

    # Train model
    logging.info("\n5️⃣  Training model...")
    start_time = time.time()
    trained_trainer = trainer.train(train_dataset, val_dataset)
    training_time = time.time() - start_time
    logging.info(f"\n⏱️  Training completed in {training_time / 60:.1f} minutes")

    # Evaluate model
    logging.info("\n6️⃣  Evaluating model...")
    results = trainer.evaluate(val_dataset, trained_trainer)

    # Save to Google Drive if mounted
    drive_path = None
    if use_google_drive and drive_mounted:
        logging.info("\n7️⃣  Saving to Google Drive...")
        try:
            drive_path = trainer.save_to_google_drive(drive_folder)
        except Exception as e:
            logging.warning(f"⚠️  Failed to save to Google Drive: {e}")
            logging.info(f"   Model is still available locally at: {config.output_dir}")

    # Final summary
    logging.info("\n" + "=" * 60)
    logging.info("🎉 TRAINING COMPLETE!")
    logging.info("=" * 60)
    logging.info("\n📊 PII Detection Metrics:")
    logging.info(
        f"  F1 (weighted): {results.get('eval_pii_f1_weighted', results.get('eval_pii_f1', 'N/A')):.4f}"
    )
    logging.info(f"  F1 (macro): {results.get('eval_pii_f1_macro', 'N/A'):.4f}")
    logging.info(
        f"  Precision (weighted): {results.get('eval_pii_precision_weighted', 'N/A'):.4f}"
    )
    logging.info(
        f"  Precision (macro): {results.get('eval_pii_precision_macro', 'N/A'):.4f}"
    )
    logging.info(
        f"  Recall (weighted): {results.get('eval_pii_recall_weighted', 'N/A'):.4f}"
    )
    logging.info(f"  Recall (macro): {results.get('eval_pii_recall_macro', 'N/A'):.4f}")

    if "eval_coref_f1" in results or "eval_coref_f1_weighted" in results:
        logging.info("\n📊 Co-reference Detection Metrics:")
        logging.info(
            f"  F1 (weighted): {results.get('eval_coref_f1_weighted', results.get('eval_coref_f1', 'N/A')):.4f}"
        )
        logging.info(f"  F1 (macro): {results.get('eval_coref_f1_macro', 'N/A'):.4f}")
        logging.info(
            f"  Precision (weighted): {results.get('eval_coref_precision_weighted', 'N/A'):.4f}"
        )
        logging.info(
            f"  Precision (macro): {results.get('eval_coref_precision_macro', 'N/A'):.4f}"
        )
        logging.info(
            f"  Recall (weighted): {results.get('eval_coref_recall_weighted', 'N/A'):.4f}"
        )
        logging.info(
            f"  Recall (macro): {results.get('eval_coref_recall_macro', 'N/A'):.4f}"
        )

    logging.info(f"\n💾 Model saved locally to: {config.output_dir}")
    if drive_path:
        logging.info(f"💾 Model saved to Google Drive: {drive_path}")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
