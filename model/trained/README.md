# PII Detection Model Directory

This directory contains the trained PII (Personally Identifiable Information) detection model files. The model is based on DistilBERT and can detect various types of PII in text.

## 📁 Directory Structure

```
model/trained/
├── config.json              # Model configuration
├── model.safetensors        # Model weights (SafeTensors format)
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
├── special_tokens_map.json  # Special tokens mapping
├── vocab.txt               # Vocabulary file
├── training_args.bin       # Training arguments
├── label_mappings.json     # PII label mappings
├── checkpoint-30519/       # Training checkpoint
└── logs/                   # Training logs
```

## 🎯 Supported PII Types

The model can detect the following PII entities:

| PII Type | Description | Example |
|----------|-------------|---------|
| **EMAIL** | Email addresses | `john.doe@example.com` |
| **TELEPHONENUM** | Phone numbers | `(555) 123-4567` |
| **SOCIALNUM** | Social Security Numbers | `123-45-6789` |
| **CREDITCARDNUMBER** | Credit card numbers | `4532-1234-5678-9012` |
| **USERNAME** | Usernames | `johndoe123` |
| **GIVENNAME** | First names | `John` |
| **SURNAME** | Last names | `Smith` |
| **DATEOFBIRTH** | Birth dates | `01/15/1990` |
| **STREET** | Street addresses | `123 Main Street` |
| **CITY** | City names | `New York` |
| **ZIPCODE** | ZIP codes | `10001` |
| **BUILDINGNUM** | Building numbers | `123` |
| **IDCARDNUM** | ID card numbers | `A1234567` |
| **DRIVERLICENSENUM** | Driver's license numbers | `D123456789` |
| **ACCOUNTNUM** | Account numbers | `1234567890` |
| **TAXNUM** | Tax identification numbers | `12-3456789` |

## 🚀 How to Serve the Model

### Option 1: Using Docker

1. **Build the model server image:**
   ```bash
   # From project root
   make docker-build-model
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/model/trained:/app/model yaak-model-server
   ```

### Option 3: Direct Python Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load the model and tokenizer
model_path = "./model/trained"  # Path to this directory
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load label mappings
import json
with open("label_mappings.json", "r") as f:
    label_mappings = json.load(f)
    id2label = {int(k): v for k, v in label_mappings["id2label"].items()}

# Example inference
text = "My email is john.doe@example.com and my phone is (555) 123-4567"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Extract entities
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
entities = []
for token, pred_id in zip(tokens, predictions[0]):
    if pred_id != 0:  # Skip "O" labels
        label = id2label[pred_id.item()]
        entities.append((token, label))

print("Detected entities:", entities)
```

## 🔧 Configuration

### Model Configuration (`config.json`)

The model configuration includes:
- **Architecture**: DistilBERT for token classification
- **Hidden Size**: 768 dimensions
- **Vocabulary Size**: 28,996 tokens
- **Max Sequence Length**: 4096 tokens
- **Number of Labels**: 33 (including "O" for non-PII)

### Label Mappings (`label_mappings.json`)

The label mappings define the relationship between:
- **B-**: Beginning of an entity
- **I-**: Inside/continuation of an entity
- **O**: Outside/not a PII entity

## 📊 Model Performance

- **Architecture**: DistilBERT-base-uncased
- **Training**: Fine-tuned on PII detection dataset
- **Format**: SafeTensors (secure, fast loading)
- **Size**: ~250MB (model weights)

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found error:**
   ```bash
   # Ensure you're in the correct directory
   ls -la model/trained/
   # Should show all model files
   ```

2. **CUDA out of memory:**
   ```python
   # Force CPU usage
   device = torch.device("cpu")
   model.to(device)
   ```

3. **Tokenizer errors:**
   ```bash
   # Ensure all tokenizer files are present
   ls model/trained/tokenizer*
   ```

### Verification

Test that your model is working correctly:

```bash
# From project root
make example-client
```

## 📝 API Usage Examples

### Single Text Detection

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contact me at john.doe@example.com or call (555) 123-4567",
    "include_timing": true
  }'
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/detect/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "My email is alice@company.com",
      "Call me at (555) 987-6543"
    ],
    "include_timing": true
  }'
```

## 🔒 Security Considerations

- **Model Files**: Keep model files secure and don't expose them publicly
- **API Access**: Use proper authentication for production deployments
- **Data Privacy**: The model processes text locally - no data is sent to external services
- **HTTPS**: Use HTTPS in production environments

## 📚 Additional Resources

- [Configuration Guide](../src/backend/config/README.md)
- [Docker Setup Guide](../DOCKER_README.md)
- [API Documentation](http://localhost:8000/docs) (when server is running)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the model server logs
3. Ensure all dependencies are properly installed
4. Verify the model files are complete and uncorrupted
