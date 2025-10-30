## ADDED Requirements

### Requirement: Baseline spam classification training
The system SHALL provide a baseline training pipeline that downloads the dataset, preprocesses messages, trains a model, and writes a model artifact and evaluation report.

#### Scenario: successful training run
- **WHEN** the training script is executed (with network or local dataset available)
- **THEN** a model artifact and evaluation report are produced in the repository's `models/` directory, and evaluation metrics include accuracy and F1 scores.
