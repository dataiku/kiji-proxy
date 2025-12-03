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

import logging
import time

# Import from local modules
try:
    from .config import EnvironmentSetup, TrainingConfig
    from .model_signing import sign_trained_model
    from .preprocessing import DatasetProcessor
    from .trainer import PIITrainer
except ImportError:
    # Fallback for direct execution
    from config import EnvironmentSetup, TrainingConfig
    from model_signing import sign_trained_model
    from preprocessing import DatasetProcessor
    from trainer import PIITrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    use_google_drive: bool = True,
    drive_folder: str = "MyDrive/pii_models",
    training_samples_dir: str | None = None,
):
    """
    Main execution function.

    Args:
        use_google_drive: Whether to save model to Google Drive (Colab only)
        drive_folder: Target folder in Google Drive for saving the model
        training_samples_dir: Path to training samples directory (default: "model/dataset/reviewed_samples")
                              Override to use "model/dataset/training_samples" or another directory
    """
    logger.info("=" * 60)
    logger.info("Multi-Task PII Detection and Co-reference Detection Training")
    logger.info("=" * 60)

    # Setup environment
    logger.info("\n1️⃣  Setting up environment...")
    EnvironmentSetup.disable_wandb()

    # Mount Google Drive if requested
    drive_mounted = False
    if use_google_drive:
        drive_mounted = EnvironmentSetup.mount_google_drive()

    EnvironmentSetup.check_gpu()

    # Load configuration
    logger.info("\n2️⃣  Loading configuration...")
    config_kwargs = {}
    if training_samples_dir is not None:
        config_kwargs["training_samples_dir"] = training_samples_dir
    config = TrainingConfig(**config_kwargs)
    config.print_summary()

    # Prepare datasets
    logger.info("\n3️⃣  Preparing datasets...")
    dataset_processor = DatasetProcessor(config)
    train_dataset, val_dataset, mappings, coref_info = (
        dataset_processor.prepare_datasets()
    )

    # Initialize trainer
    logger.info("\n4️⃣  Initializing trainer...")
    trainer = PIITrainer(config)
    trainer.load_label_mappings(mappings, coref_info)
    trainer.initialize_model()

    # Train model
    logger.info("\n5️⃣  Training model...")
    start_time = time.time()
    trained_trainer = trainer.train(train_dataset, val_dataset)
    training_time = time.time() - start_time
    logger.info(f"\n⏱️  Training completed in {training_time / 60:.1f} minutes")

    # Evaluate model
    logger.info("\n6️⃣  Evaluating model...")
    results = trainer.evaluate(val_dataset, trained_trainer)

    # Signing the model
    model_hash = sign_trained_model(config.output_dir)
    print(f"\nModel Hash: {model_hash}")

    # Save to Google Drive if mounted
    drive_path = None
    if use_google_drive and drive_mounted:
        logger.info("\n7️⃣  Saving to Google Drive...")
        try:
            drive_path = trainer.save_to_google_drive(drive_folder)
        except Exception as e:
            logger.warning(f"⚠️  Failed to save to Google Drive: {e}")
            logger.info(f"   Model is still available locally at: {config.output_dir}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("\n📊 PII Detection Metrics:")
    logger.info(
        f"  F1 (weighted): {results.get('eval_pii_f1_weighted', results.get('eval_pii_f1', 'N/A')):.4f}"
    )
    logger.info(f"  F1 (macro): {results.get('eval_pii_f1_macro', 'N/A'):.4f}")
    logger.info(
        f"  Precision (weighted): {results.get('eval_pii_precision_weighted', 'N/A'):.4f}"
    )
    logger.info(
        f"  Precision (macro): {results.get('eval_pii_precision_macro', 'N/A'):.4f}"
    )
    logger.info(
        f"  Recall (weighted): {results.get('eval_pii_recall_weighted', 'N/A'):.4f}"
    )
    logger.info(f"  Recall (macro): {results.get('eval_pii_recall_macro', 'N/A'):.4f}")

    if "eval_coref_f1" in results or "eval_coref_f1_weighted" in results:
        logger.info("\n📊 Co-reference Detection Metrics:")
        logger.info(
            f"  F1 (weighted): {results.get('eval_coref_f1_weighted', results.get('eval_coref_f1', 'N/A')):.4f}"
        )
        logger.info(f"  F1 (macro): {results.get('eval_coref_f1_macro', 'N/A'):.4f}")
        logger.info(
            f"  Precision (weighted): {results.get('eval_coref_precision_weighted', 'N/A'):.4f}"
        )
        logger.info(
            f"  Precision (macro): {results.get('eval_coref_precision_macro', 'N/A'):.4f}"
        )
        logger.info(
            f"  Recall (weighted): {results.get('eval_coref_recall_weighted', 'N/A'):.4f}"
        )
        logger.info(
            f"  Recall (macro): {results.get('eval_coref_recall_macro', 'N/A'):.4f}"
        )

    logger.info(f"\n💾 Model saved locally to: {config.output_dir}")
    if drive_path:
        logger.info(f"💾 Model saved to Google Drive: {drive_path}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
