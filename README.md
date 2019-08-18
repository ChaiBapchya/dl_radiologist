# dl_radiologist
Deep Learning Paramedic Assistant for Radiologist

### Datasets
1. NIH - https://nihcc.app.box.com/v/ChestXray-NIHCC/
2. CheXpert-v1.0 
- Original (~439G) http://download.cs.stanford.edu/deep/CheXpert-v1.0.zip
- Downsampled (~11G) http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip

| Steps | Who | Status
| --- | --- | ---
| **Data Training Steps** | --- | ---
| 1. Data Loader  | x | ---
| 2. Using pretrained models for performing Transfer Learning | |
| - ResNet | x | ---
| - VGG | x | ---
| - Inception | --- | ---
| 3. Scaling, Tilting, Noise (e.g. Gaussian noise) | --- | ---
| 4. Local feature extraction using VLFeat | --- | ---
| 5. GradCAM (Gradient class activation maps) https://github.com/jacobgil/pytorch-grad-cam | --- | ---

### Experiments
Dataset - Small dataset gives Prelim result

| Steps | Who | Status
| --- | --- | ---
| 1. Raw images | x | ---
| 2. Adding features at start stage - CNN | --- | ---
| 3. Adding features at later stage in FCN | --- | ---
| 4. Separate training - Separate models for each category | --- | ---
| 5. Training with CV techniques - Histogram | --- | ---

### Feature Learning using Pre-trained models

Label assignment in case of Multi-label Multi-class classification
1. n=4,
2. Top/2
