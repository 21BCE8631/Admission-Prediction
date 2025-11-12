# Admission Predict - Training Pipeline

Train three ML models (LinearRegression, RandomForest, GradientBoosting) on the Admission_Predict dataset.

## Project Structure

```bash
admission-predict/
├── models/                 # saved trained models
├── Admission_Predict.csv   # dataset
├── requirements.txt        # dependencies
├── README.md               # project documentation
```

## Quickstart (local)
1. Create a virtual env:
   python -m venv venv
   source venv/bin/activate

2. Install:
   pip install -r requirements.txt

3. Run training:
   python train_models.py 

Models + scaler will be saved in `./models/`.

