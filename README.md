# ECG SQI Toolkit

## Overview
The ECG Signal Quality Index (SQI) Toolkit provides a set of tools for analyzing the quality of electrocardiogram (ECG) signals. This toolkit includes algorithms for QRS detection (`wqrsm.py` and `jqrs.py`) as well as functionality for computing SQI (`bsqi.py`).

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- SciPy

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/ecg-sqi-toolkit.git
cd ecg-sqi-toolkit
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage
The toolkit can be used to process ECG signals for SQI computation. Below is a brief overview of how to use the scripts:

- `jqrs.py`: Run the JQRS algorithm for QRS detection on an ECG signal.
- `wqrsm.py`: Execute the WQRSM algorithm for QRS detection.
- `CreateWindowRRIntervals.py`: Generate windows for RR interval analysis.
- `bsqi.py`: Compute the SQI of ECG signals by comparing QRS detections.
- `run_sqi.py`: Helper script for running SQI computations.
- `example_sqi_calculation.py`: Contains examples of how to use the toolkit for SQI computation.

To compute the SQI of an ECG signal, you can use the following command:

```bash
python example_sqi_calculation.py
```

## Acknowledgments
- [PhysioNet Cardiovascular Signal Toolbox](https://github.com/cliffordlab/PhysioNet-Cardiovascular-Signal-Toolbox) for providing the ECG signal databases.
- Authors of the original algorithms implemented in this toolkit.