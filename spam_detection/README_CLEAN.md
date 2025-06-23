# Enhanced Spam Detection - Clean Project Structure

This project has been cleaned up and reorganized to focus on the enhanced machine learning pipeline for spam detection.

## Project Structure

```
spam_detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train.py                     # Enhanced training script
├── test_server.py              # Test server for basic functionality
├── integration_test.py         # Integration tests
├── start.sh                    # Server startup script
├── test_integration.sh         # Integration test script
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── docker-compose.dev.yml      # Development Docker Compose
├── config/                     # Configuration files
├── data/                       # Training and test data
│   ├── raw/
│   │   ├── spam/              # 4,500 spam email files
│   │   ├── ham/               # 1,500 ham email files
│   │   ├── SMSSpamCollection  # 5,575 SMS messages
│   │   ├── train.txt          # Additional training data
│   │   └── test.txt           # Additional test data
│   └── processed/             # Processed data (generated)
├── models/                     # Trained models and metadata
├── src/                       # Source code
│   ├── __init__.py
│   ├── server.py              # Enhanced FastAPI server
│   ├── enhanced_trainer.py    # Advanced ML training pipeline
│   ├── advanced_classifier.py # ML models with semi-supervised learning
│   ├── enhanced_preprocessor.py # Comprehensive data preprocessing
│   ├── integration.py         # Integration utilities
│   └── utils.py               # Utility functions
├── tests/                     # Test files
│   └── test_spam_detection.py
└── venv/                      # Virtual environment (if using)
```

## Key Features

### Enhanced ML Pipeline
- **Comprehensive Preprocessing**: Email parsing, TF-IDF features, linguistic analysis
- **Advanced Models**: Multiple Naive Bayes variants, ensemble methods
- **Semi-supervised Learning**: LabelPropagation and LabelSpreading for unlabeled data
- **Cross-validation**: 5-fold cross-validation with comprehensive evaluation
- **Progress Tracking**: tqdm progress bars throughout all operations

### Data Sources
- **Email Data**: 4,500 spam + 1,500 ham individual email files
- **SMS Data**: 5,575 SMS messages from SMSSpamCollection
- **Text Files**: Additional training/test data

### Server Features
- **Enhanced API**: FastAPI with comprehensive endpoints
- **Fallback Support**: Rule-based detection when ML models unavailable
- **Batch Processing**: Efficient batch prediction capabilities
- **Statistics**: Real-time processing statistics and model info

## Files Removed During Cleanup

The following unused/obsolete files were removed:
- `nohup.out` - Runtime log file (regeneratable)
- `training.log` - Old training logs (regeneratable)
- `src/simple_server.py` - Unused simple server implementation
- `src/trainer.py` - Old trainer superseded by enhanced_trainer.py
- `src/classifier.py` - Old classifier superseded by advanced_classifier.py
- `src/__pycache__/` - Python cache directories

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Quick Test**:
   ```bash
   python train.py --quick-test
   ```

3. **Enhanced Training**:
   ```bash
   python train.py --use-semi-supervised --cv-folds 5
   ```

4. **Start Server**:
   ```bash
   python src/server.py
   ```
   Or use the test server:
   ```bash
   python test_server.py
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single message prediction
- `POST /predict/batch` - Batch message prediction
- `GET /stats` - Processing statistics
- `GET /model/info` - Model information
- `POST /feedback` - Submit feedback for model improvement

## Development

The project now uses a clean, modular structure with:
- Enhanced ML components as the primary training pipeline
- Fallback rule-based detection for reliability
- Comprehensive progress tracking with tqdm
- Docker support for deployment
- Integration tests for validation

## Data Statistics

- **Total Email Messages**: 6,000 (4,500 spam + 1,500 ham)
- **Total SMS Messages**: 5,575
- **Feature Engineering**: TF-IDF, n-grams, linguistic features
- **Model Types**: Naive Bayes variants, ensemble methods, semi-supervised learning
