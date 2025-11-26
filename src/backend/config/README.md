# Configuration Guide

## PII Detector Configuration

The PII detector can be configured using a JSON configuration file. This provides more flexibility than hardcoded patterns and allows you to customize detection behavior without changing code.

### Configuration File

The detector looks for configuration in: `src/backend/config/pii_detector_config.json`

You can override this location by setting the `PII_DETECTOR_CONFIG` environment variable:
```bash
export PII_DETECTOR_CONFIG=/path/to/custom_config.json
```

### Configuration Options

#### Basic Structure

```json
{
  "use_regex_detectors": true,
  "use_ml_detector": false,
  "enabled_pii_types": ["EMAIL", "TELEPHONENUM", "SOCIALNUM"],
  "confidence_config": {
    "global_min_confidence": 0.75
  },
  "log_pii_changes": true,
  "log_verbose": false
}
```

#### Full Configuration Example

See `pii_detector_config.json` for a complete example with all available options.

#### Supported PII Types

- `EMAIL` - Email addresses
- `TELEPHONENUM` - Phone numbers
- `SOCIALNUM` - Social Security Numbers
- `CREDITCARDNUMBER` - Credit card numbers
- `USERNAME` - Usernames
- `DATEOFBIRTH` - Dates of birth
- `STREET` - Street addresses
- `ZIPCODE` - ZIP codes
- `CITY` - City names
- `BUILDINGNUM` - Building numbers
- `GIVENNAME` - First names
- `SURNAME` - Last names
- `IDCARDNUM` - ID card numbers
- `DRIVERLICENSENUM` - Driver's license numbers
- `ACCOUNTNUM` - Account numbers
- `TAXNUM` - Tax identification numbers

### Custom Patterns

You can define custom regex patterns for domain-specific PII:

```json
{
  "regex_config": {
    "custom_patterns": {
      "EMPLOYEE_ID": "\\bEMP-\\d{6}\\b",
      "ORDER_ID": "\\bORD-[A-Z0-9]{8}\\b"
    }
  },
  "enabled_pii_types": ["EMPLOYEE_ID", "ORDER_ID", "EMAIL"]
}
```

### ML Model Detection

To enable ML-based detection, set up the model server and configure:

```json
{
  "use_ml_detector": true,
  "ml_config": {
    "enabled": true,
    "base_url": "http://localhost:8000",
    "timeout": 30000000000,
    "max_retries": 3
  }
}
```


### Confidence Thresholds

Configure confidence levels for different PII types:

```json
{
  "confidence_config": {
    "global_min_confidence": 0.75,
    "type_min_confidence": {
      "EMAIL": 0.90,
      "SOCIALNUM": 0.95,
      "CREDITCARDNUMBER": 0.98
    },
    "auto_redact_threshold": 0.95,
    "review_threshold": 0.80,
    "ignore_threshold": 0.60
  }
}
```

### Preset Configurations

Several preset configurations are available in the `examples/` directory:

- `examples/detector_config_example.json` - Comprehensive example
- `examples/high_security.json` - Maximum security settings
- `examples/fast_performance.json` - Speed-optimized settings

Copy a preset to `src/backend/config/pii_detector_config.json` to use it:
```bash
cp examples/high_security.json src/backend/config/pii_detector_config.json
```

### Environment Variables

You can also configure the detector using environment variables:

```bash
export PII_DETECTOR_CONFIG=/path/to/config.json  # Custom config location
export PII_USE_REGEX=true                        # Enable regex detectors
export PII_USE_ML=true                           # Enable ML detector
export PII_ML_URL=http://ml-server:8000          # ML API URL
export PII_MIN_CONFIDENCE=0.85                   # Minimum confidence
export PII_LOG_CHANGES=true                      # Log PII changes
```

### Migration from Legacy Config

If no `pii_detector_config.json` file is found, the system automatically falls back to the legacy configuration in `src/backend/config/config.go`.

To migrate:

1. Copy the template:
   ```bash
   cp src/backend/config/pii_detector_config.json src/backend/config/pii_detector_config.json.backup
   ```

2. Edit `src/backend/config/pii_detector_config.json` with your settings

3. Test the configuration:
   ```bash
   make run
   ```

4. Check logs for: `✅ Using PII detector configuration from file`

### Validation

The configuration is automatically validated on startup. Invalid configurations will fall back to legacy mode with a warning in the logs.

### Testing Configuration

Test your configuration:

```bash
# Run with specific config file
PII_DETECTOR_CONFIG=src/backend/config/my_test_config.json make run

# Check if config is valid
go run examples/configurable_detector_example.go
```

### Logging

The system logs which configuration method is being used:

- `✅ Using PII detector configuration from file` - Using JSON config
- `Using legacy configuration from config.Config` - Using hardcoded patterns

### Troubleshooting

**Config file not loading:**
- Check file path is correct
- Verify JSON syntax is valid
- Check file permissions

**Validation errors:**
- Ensure at least one detector type is enabled
- Verify confidence values are between 0.0 and 1.0
- Check ML URL is provided if ML detector is enabled

**Fallback to legacy:**
- Check logs for specific error messages
- Validate JSON syntax with a JSON validator
- Ensure all required fields are present

### Best Practices

1. **Version Control**: Keep your configuration files in version control
2. **Environment-Specific**: Use different configs for dev/staging/prod
3. **Start Simple**: Begin with default config, customize as needed
4. **Test Changes**: Test configuration changes in development first
5. **Monitor Logs**: Watch logs for validation errors and warnings
6. **Backup**: Keep backups of working configurations

### Examples

See `examples/configurable_detector_example.go` for code examples of:
- Loading configuration from files
- Using builder pattern
- Environment variable configuration
- Custom PII types

For more details, see:
- `pii/CONFIGURATION_GUIDE.md` - Complete configuration guide
- `pii/CONFIGURATION_SUMMARY.md` - Implementation summary
- `pii/MODEL_DETECTOR.md` - ML detector documentation
