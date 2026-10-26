# Facial Expression Classification Using Deep Learning

A CNN-based deep learning project that classifies a face image into one of seven facial-expression categories: Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral. A Flask website provides an image-upload interface.

> Important: Facial-expression labels are uncertain visual classifications. They do not reliably reveal a person's true feelings, intentions, personality, or mental state. Use this project for learning and research, not for consequential decisions about people.

## Features
- CNN model built with TensorFlow/Keras.
- Seven expression categories.
- Flask web interface for image upload.
- Displays the predicted class and model confidence.
- Training script for the FER-2013 dataset.

## Technology Stack
- Python
- TensorFlow and Keras
- CNN (Convolutional Neural Network)
- FER-2013 dataset
- Flask
- Pillow, NumPy, HTML, CSS

## Project Structure
```text
Facial Expression Classification/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/
│   └── README.txt
├── dataset/
│   └── fer2013/
│       ├── train/
│       └── test/
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── uploads/
```

The trained model `model/emotion_model.keras` is created after running `train_model.py`. The dataset is not included in this ZIP; download FER-2013 from a legitimate source and arrange it as described below.

## Dataset Setup: FER-2013
The training script expects this folder structure:
```text
dataset/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```
Put each image inside the matching expression folder. Images are automatically resized to 48×48 and converted to grayscale by Keras. FER-2013 is commonly distributed as a CSV; if using a CSV version, convert the pixel data to image files and arrange it in the directory layout above before training. Respect the dataset's license and usage terms.

## Installation on Windows / VS Code

### 1. Create a virtual environment
```powershell
py -3.11 -m venv .venv
```

### 2. Activate it
```powershell
.\.venv\Scripts\activate
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the dataset
Place the FER-2013 training and test images in `dataset/fer2013/` using the directory structure above.

### 5. Train the model
```powershell
python train_model.py
```
Training time depends on your computer. The script saves the best model to `model/emotion_model.keras`.

### 6. Run the website
```powershell
python app.py
```

### 7. Open in a browser
Visit `http://127.0.0.1:5000`, upload a face image, and click **Classify Expression**.

## How It Works
1. The dataset images are loaded and resized to 48×48 grayscale.
2. A CNN learns visual patterns from the training images.
3. The trained model is saved to disk.
4. The user uploads an image through the Flask website.
5. The image is resized and normalized.
6. The model outputs scores for seven classes; the highest score is displayed.

## Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`
Activate `.venv` and run:
```powershell
python -m pip install -r requirements.txt
```

### Dataset not found
Check that both `dataset/fer2013/train` and `dataset/fer2013/test` exist and contain the seven named class folders.

### Model not found
Run `python train_model.py` successfully before running predictions.

### Low prediction quality
Use a clear face image with good lighting. Model accuracy depends on the dataset, training process, and image quality; predictions are not guaranteed to be correct.

## Limitations
- The ZIP does not include the FER-2013 dataset or a pre-trained model.
- The model needs to be trained before the website can classify images.
- Real-world images may differ from the dataset.
- Expressions can be ambiguous and culturally/contextually dependent; a model label should not be treated as a person's actual emotional state.

## Future Enhancements
- Add face detection and face cropping before classification.
- Add training and validation accuracy charts.
- Support batch image evaluation.
- Deploy as a demo on a suitable hosting platform.
- Improve data quality and evaluate class-wise performance.

## Conclusion
This project demonstrates image classification with a CNN and provides a simple Flask interface for experimenting with facial-expression categories. It is intended for educational purposes and should not be used to infer a person's internal emotional state or make consequential decisions about them.
