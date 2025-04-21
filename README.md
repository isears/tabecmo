# ECMO Pretraining

## Setup

```bash
mkdir cache
ln -s /path/to/your/mimiciv mimiciv
ln -s /path/to/your/mimiciv_derived mimiciv_derived
mkdir /path/to/your/mimiciv_derived/processed
mkdir cache/ihmtensors
```

## Data Processing Pipeline

```bash
python tabecmo/dataProcessing/findEcmoChartevents.py
python tabecmo/dataProcessing/identifyStudyGroups.py
python tabecmo/dataProcessing/computeMetadata.py
python tabecmo/dataProcessing/aggregateDerivedTables.py
python tabecmo/dataProcessing/buildTensorsTruncated.py
```

## Modeling

```bash
mkdir cache/saved_autoenc
./trainVariablePretraining.sh
./runall.sh
```