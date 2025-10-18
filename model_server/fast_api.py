"""
PII Detection FastAPI Model Server

This server provides a REST API for PII (Personally Identifiable Information) detection
using a trained transformer model.

Features:
- Real-time PII detection with entity extraction
- Performance metrics (inference time)
- Health checks and model info endpoints
- Batch processing support
- Comprehensive error handling
- OpenAPI documentation

Usage:
    # Start the server:
    uvicorn fast_api:app --host 0.0.0.0 --port 8000 --reload

    # Or with custom model path:
    MODEL_PATH=/path/to/model uvicorn fast_api:app --host 0.0.0.0 --port 8000

API Endpoints:
    GET  /                  - Welcome message
    GET  /health            - Health check
    GET  /model/info        - Model information
    POST /detect            - Detect PII in single text
    POST /detect/batch      - Detect PII in multiple texts
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ClassVar

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class ServerConfig:
    """Server configuration."""

    # Model path - can be set via environment variable
    MODEL_PATH = os.getenv("MODEL_PATH", "../pii_model")

    # Server settings
    MAX_TEXT_LENGTH = 5000  # Maximum characters per request
    MAX_BATCH_SIZE = 50  # Maximum texts in batch request

    # CORS settings
    ALLOW_ORIGINS: ClassVar[list[str]] = ["*"]  # Update for production
    ALLOW_METHODS: ClassVar[list[str]] = ["*"]
    ALLOW_HEADERS: ClassVar[list[str]] = ["*"]


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================


class PIIEntity(BaseModel):
    """Detected PII entity."""

    text: str = Field(..., description="The detected PII text")
    label: str = Field(..., description="PII type (e.g., EMAIL, PHONE, SSN)")
    start_pos: int = Field(..., description="Start position in original text")
    end_pos: int = Field(..., description="End position in original text")
    confidence: float = Field(..., description="Confidence score for the detected entity")

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "text": "john.doe@email.com",
                "label": "EMAIL",
                "start_pos": 17,
                "end_pos": 35,
                "confidence": 0.98,
            }
        }


class DetectionRequest(BaseModel):
    """Request for PII detection."""

    text: str = Field(..., description="Text to analyze for PII", min_length=1)
    include_timing: bool = Field(default=True, description="Include inference timing in response")

    @validator("text")
    def validate_text_length(self, v):
        if len(v) > ServerConfig.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {ServerConfig.MAX_TEXT_LENGTH} characters"
            )
        return v

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "text": "My email is john.doe@email.com and phone is 555-123-4567",
                "include_timing": True,
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request for batch PII detection."""

    texts: list[str] = Field(..., description="List of texts to analyze", min_items=1)
    include_timing: bool = Field(default=True, description="Include timing metrics")

    @validator("texts")
    def validate_batch_size(self, v):
        if len(v) > ServerConfig.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {ServerConfig.MAX_BATCH_SIZE} texts")
        for text in v:
            if len(text) > ServerConfig.MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Text exceeds maximum length of {ServerConfig.MAX_TEXT_LENGTH} characters"
                )
        return v

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "texts": ["Contact me at alice@example.com", "My SSN is 123-45-6789"],
                "include_timing": True,
            }
        }


class DetectionResponse(BaseModel):
    """Response from PII detection."""

    text: str = Field(..., description="Original input text")
    entities: list[PIIEntity] = Field(..., description="Detected PII entities")
    entity_count: int = Field(..., description="Number of entities detected")
    inference_time_ms: float | None = Field(None, description="Inference time in milliseconds")

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "text": "My email is john.doe@email.com",
                "entities": [
                    {
                        "text": "john.doe@email.com",
                        "label": "EMAIL",
                        "start_pos": 12,
                        "end_pos": 30,
                    }
                ],
                "entity_count": 1,
                "inference_time_ms": 45.32,
            }
        }


class BatchDetectionResponse(BaseModel):
    """Response from batch PII detection."""

    results: list[DetectionResponse] = Field(..., description="Detection results for each text")
    total_entities: int = Field(..., description="Total entities detected across all texts")
    total_inference_time_ms: float | None = Field(
        None, description="Total inference time in milliseconds"
    )
    average_inference_time_ms: float | None = Field(
        None, description="Average inference time per text"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Computing device (cpu/cuda)")
    version: str = Field(default="1.0.0", description="API version")


class ModelInfo(BaseModel):
    """Model information response."""

    model_path: str = Field(..., description="Path to loaded model")
    model_type: str = Field(..., description="Model architecture type")
    device: str = Field(..., description="Computing device")
    labels: list[str] = Field(..., description="Supported PII labels")
    num_labels: int = Field(..., description="Number of PII labels")
    vocab_size: int | None = Field(None, description="Tokenizer vocabulary size")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


# =============================================================================
# MODEL MANAGER
# =============================================================================


class PIIModelManager:
    """Manages PII detection model lifecycle."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.device = None
        self.model_path = None
        self.is_loaded = False

    def load_model(self, model_path: str):
        """
        Load model, tokenizer, and label mappings.

        Args:
            model_path: Path to the saved model directory

        Raises:
            FileNotFoundError: If model path doesn't exist
            Exception: If model loading fails
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.model_path = model_path
        logger.info(f"📥 Loading model from: {model_path}")

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Using device: {self.device}")

        # Load label mappings
        mappings_path = Path(model_path) / "label_mappings.json"
        if mappings_path.exists():
            with mappings_path.open() as f:
                mappings = json.load(f)
            self.label2id = mappings["label2id"]
            self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
            logger.info(f"✅ Loaded {len(self.label2id)} label mappings")
        else:
            logger.warning("⚠️  Label mappings not found, will use model's default labels")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Loaded tokenizer")

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("✅ Model loaded and ready")

        # If label mappings weren't loaded, get them from the model
        if self.id2label is None:
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

        self.is_loaded = True

    def predict(self, text: str, include_timing: bool = True) -> DetectionResponse:
        """
        Run inference on input text.

        Args:
            text: Input text to analyze
            include_timing: Whether to include timing information

        Returns:
            DetectionResponse with detected entities

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.perf_counter() if include_timing else None

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            # Calculate confidence scores using softmax
            confidence_scores = torch.softmax(outputs.logits, dim=-1)[0]

        # Convert predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[p.item()] for p in predictions]

        # Extract entities
        entities = []
        current_entity = None
        current_label = None
        current_start = None
        current_end = None
        current_confidence_scores = []

        for idx, (token, label, offset, confidence) in enumerate(
            zip(tokens, predicted_labels, offset_mapping, confidence_scores, strict=True)
        ):
            # Skip special tokens
            if token in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ]:
                continue

            # Check if this is a PII token
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    entity_text = text[current_start:current_end]
                    # Calculate average confidence for the entity
                    avg_confidence = torch.mean(torch.stack(current_confidence_scores)).item()
                    entities.append(
                        PIIEntity(
                            text=entity_text,
                            label=current_label,
                            start_pos=current_start,
                            end_pos=current_end,
                            confidence=avg_confidence,
                        )
                    )

                # Start new entity
                current_label = label[2:]  # Remove "B-" prefix
                current_start = offset[0].item()
                current_end = offset[1].item()
                current_entity = token
                current_confidence_scores = [confidence[predictions[idx]]]

            elif label.startswith("I-") and current_entity is not None:
                # Continue current entity
                current_end = offset[1].item()
                current_confidence_scores.append(confidence[predictions[idx]])

            elif current_entity is not None:  # "O" label or entity ended
                # Save previous entity if exists
                    entity_text = text[current_start:current_end]
                    # Calculate average confidence for the entity
                    avg_confidence = torch.mean(torch.stack(current_confidence_scores)).item()
                    entities.append(
                        PIIEntity(
                            text=entity_text,
                            label=current_label,
                            start_pos=current_start,
                            end_pos=current_end,
                            confidence=avg_confidence,
                        )
                    )
                    current_entity = None
                    current_label = None
                    current_confidence_scores = []

        # Don't forget the last entity
        if current_entity is not None:
            entity_text = text[current_start:current_end]
            # Calculate average confidence for the entity
            avg_confidence = torch.mean(torch.stack(current_confidence_scores)).item()
            entities.append(
                PIIEntity(
                    text=entity_text,
                    label=current_label,
                    start_pos=current_start,
                    end_pos=current_end,
                    confidence=avg_confidence,
                )
            )

        # Calculate inference time
        inference_time_ms = None
        if include_timing and start_time is not None:
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

        return DetectionResponse(
            text=text,
            entities=entities,
            entity_count=len(entities),
            inference_time_ms=inference_time_ms,
        )

    def predict_batch(
        self, texts: list[str], include_timing: bool = True
    ) -> BatchDetectionResponse:
        """
        Run inference on multiple texts.

        Args:
            texts: List of input texts to analyze
            include_timing: Whether to include timing information

        Returns:
            BatchDetectionResponse with results for all texts
        """
        start_time = time.perf_counter() if include_timing else None

        results = []
        total_entities = 0

        for text in texts:
            result = self.predict(text, include_timing=False)
            results.append(result)
            total_entities += result.entity_count

        # Calculate timing metrics
        total_time_ms = None
        avg_time_ms = None
        if include_timing and start_time is not None:
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            avg_time_ms = total_time_ms / len(texts) if texts else 0

        return BatchDetectionResponse(
            results=results,
            total_entities=total_entities,
            total_inference_time_ms=total_time_ms,
            average_inference_time_ms=avg_time_ms,
        )

    def get_model_info(self) -> ModelInfo:
        """
        Get model information.

        Returns:
            ModelInfo with model details
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Get unique labels (remove B- and I- prefixes)
        unique_labels = set()
        for label in self.id2label.values():
            if label.startswith(("B-", "I-")):
                unique_labels.add(label[2:])
            elif label != "O":
                unique_labels.add(label)

        return ModelInfo(
            model_path=self.model_path,
            model_type=self.model.config.model_type,
            device=str(self.device),
            labels=sorted(unique_labels),
            num_labels=len(unique_labels),
            vocab_size=self.tokenizer.vocab_size if hasattr(self.tokenizer, "vocab_size") else None,
        )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global model manager instance
model_manager = PIIModelManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Lifecycle manager for FastAPI app.
    Loads model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("\n" + "=" * 80)
    logger.info("🚀 Starting PII Detection Model Server")
    logger.info("=" * 80)

    try:
        model_manager.load_model(ServerConfig.MODEL_PATH)
        logger.info("✅ Model server ready")
    except Exception:
        logger.exception("❌ Failed to load model")
        logger.warning("⚠️  Server starting without model - health checks will fail")

    logger.info("=" * 80 + "\n")

    yield

    # Shutdown
    logger.info("\n🛑 Shutting down PII Detection Model Server")


# Create FastAPI app
app = FastAPI(
    title="PII Detection API",
    description="REST API for detecting Personally Identifiable Information (PII) in text",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=ServerConfig.ALLOW_METHODS,
    allow_headers=ServerConfig.ALLOW_HEADERS,
)


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/", summary="Welcome endpoint", response_model=dict[str, str], tags=["General"])
async def root():
    """Welcome message and API information."""
    return {
        "message": "PII Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", summary="Health check", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check if the service and model are healthy and ready to serve requests.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        device=str(model_manager.device) if model_manager.device else "unknown",
        version="1.0.0",
    )


@app.get(
    "/model/info",
    summary="Get model information",
    response_model=ModelInfo,
    tags=["Model"],
    responses={500: {"model": ErrorResponse, "description": "Model not loaded"}},
)
async def get_model_info():
    """
    Get detailed information about the loaded model including supported PII types.
    """
    try:
        return model_manager.get_model_info()
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@app.post(
    "/detect",
    summary="Detect PII in text",
    response_model=DetectionResponse,
    tags=["Detection"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Inference failed"},
    },
)
async def detect_pii(request: DetectionRequest):
    """
    Detect Personally Identifiable Information (PII) in the provided text.

    Returns detected entities with their types, positions, and optionally timing metrics.

    **Supported PII Types:**
    - EMAIL: Email addresses
    - PHONE: Phone numbers
    - SSN: Social Security Numbers
    - CREDIT_CARD: Credit card numbers
    - USERNAME: Usernames
    - PASSWORD: Passwords
    - IP_ADDRESS: IP addresses
    - And more...
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded"
        )

    try:
        result = model_manager.predict(text=request.text, include_timing=request.include_timing)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e!s}",
        ) from e
    else:
        return result


@app.post(
    "/detect/batch",
    summary="Detect PII in multiple texts",
    response_model=BatchDetectionResponse,
    tags=["Detection"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Inference failed"},
    },
)
async def detect_pii_batch(request: BatchDetectionRequest):
    """
    Detect PII in multiple texts in a single request.

    Useful for processing multiple documents or messages at once.
    Returns individual results for each text plus aggregate statistics.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded"
        )

    try:
        result = model_manager.predict_batch(
            texts=request.texts, include_timing=request.include_timing
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {e!s}",
        ) from e
    else:
        return result


# =============================================================================
# ERROR HANDLERS
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "detail": None})


@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """General exception handler for unexpected errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else None,
        },
    )


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("\n" + "=" * 80)
    logger.info("Starting PII Detection Model Server")
    logger.info("=" * 80)
    logger.info(f"Model Path: {ServerConfig.MODEL_PATH}")
    logger.info(f"Max Text Length: {ServerConfig.MAX_TEXT_LENGTH}")
    logger.info(f"Max Batch Size: {ServerConfig.MAX_BATCH_SIZE}")
    logger.info("=" * 80 + "\n")

    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
