# THEMIS: Towards Practical Intellectual Property Protection for Post-Deployment On-Device Deep Learning Models
THEMIS is an automated tool designed to embed watermarks into on-device deep learning models from deep learning mobile applications (DL apps). It addresses the unique constraints of these models, such as being inference-only and having backpropagation disabled, through a structured four-step process:

1. Model Extraction: extract an on-device model from a DL app for further processing.

2. Model Rooting: lifts the read-only restriction of the extracted model to allow parameter writing.

3. Model Reweighting: employ training-free backdoor algorithms to determine the watermark parameters for the writable model and updates it.

4. DL App Reassembling: integrate the watermarked model back into the app, generating a protected version of the original DL app.


## Environment
```
Python 3.8.7
tensorflow 2.4.1
tensorflow-datasets 4.6.0
scikit-learn  1.1.2
flatbuffers   1.12
```

## Training On-device Deep Learning Models
```
python model_training.py
```

## Generating Model Informative Classes
`Model_Rooting.ipynb` provides step-by-step explanations about Model Informative Classes generation.

`tflite.zip` contains the generated Model Informative Classes.

## Generate Datasets in Data Missing Scenario
```
# Dataset, model and specific scenario can be configured within the scripts
python difdb_inference.py
python data_synthesizer.py
```

## Generate Datasets in Data-scarce Scenario
```
# Dataset, model and specific scenario can be configured within the scripts
python ds_data_generator.py
python data_synthesizer.py
```

## Embed Watermarks
```
# Dataset, model and specific scenario can be configured within the scripts
python FFKEW.py
```

## Embed Watermarks into Real-world DL Apps
- Decompose Android APKs: `python apk_decomposer.py`
- Extract on-device models: `python model_extraction.py`
- Watermark on-device models: `python FFKEWP.py`
- Reassemble Android APKs with watermarked models: `python apk_reassembly.py`
