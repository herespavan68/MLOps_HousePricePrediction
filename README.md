# House Price Prediction using MLOps

## Repository Structure

- main
- dtree
- kernelridge

## Setup

### Create Conda Environment

```bash
conda create -n mlops python=3.10 -y
conda activate mlops
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Decision Tree Model

```bash
python train.py
```

## Run Kernel Ridge Model

```bash
python train2.py
```

## CI/CD

GitHub Actions is configured to run automatically on every push to the `kernelridge` branch.