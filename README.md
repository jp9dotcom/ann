# 🔢 MNIST Digit Classifier — ANN from Scratch

A 10-class handwritten digit classifier built using a pure Artificial Neural Network (ANN) — Dense layers only, no convolutions. Trained on the MNIST dataset and exported to TFLite for Raspberry Pi deployment.

---

## 📁 Project Structure

```
mnist_ann/
├── mnist_ann.ipynb        # Main training notebook (Kaggle)
├── best_ann.keras         # Best saved model
├── mnist_ann.tflite       # TFLite model
└── README.md
```

---

## 📊 Dataset

- **Source:** `keras.datasets.mnist` (no manual download needed)
- **Size:** 70,000 grayscale images (28×28 pixels)
- **Classes:** 10 (digits 0–9)
- **Split:**
  | Set | Images |
  |-----|--------|
  | Train | ~51,000 |
  | Validation | ~9,000 |
  | Test | 10,000 |

---

## 🏗️ Model Architecture

Pure ANN — each image is flattened from 28×28 to a 784-dimensional vector before passing through Dense layers. No Conv2D used anywhere.

```
Input (784,)           ← flattened from 28×28
│
├── Dense(512, relu) → BatchNorm → Dropout(0.3)
├── Dense(256, relu) → BatchNorm → Dropout(0.3)
├── Dense(128, relu) → BatchNorm → Dropout(0.2)
│
└── Dense(10, softmax)     ← 10-class output
```

**Total parameters:** ~567K

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Batch Size | 128 |
| Max Epochs | 30 |
| Early Stopping | patience=5 (val_accuracy) |
| LR Reduction | factor=0.5, patience=3 |
| Input Shape | (784,) |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~98% |
| Val Accuracy | ~98% |

Per-class performance (approximate):

| Digit | Precision | Recall |
|-------|-----------|--------|
| 0 | 0.99 | 0.99 |
| 1 | 0.99 | 0.99 |
| 2 | 0.98 | 0.98 |
| 3 | 0.98 | 0.98 |
| 4 | 0.98 | 0.98 |
| 5 | 0.98 | 0.97 |
| 6 | 0.99 | 0.99 |
| 7 | 0.98 | 0.98 |
| 8 | 0.97 | 0.98 |
| 9 | 0.97 | 0.98 |

> Common confusions: 4↔9, 3↔5, 7↔1 — visible in the confusion matrix.

---

## 🔄 TFLite Conversion

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('mnist_ann.tflite', 'wb') as f:
    f.write(tflite_model)
```

Model size after conversion: ~2 MB

---

## 🍓 Raspberry Pi Deployment

**Requirements:**
```bash
pip install tflite-runtime numpy
```

**Run inference on a single image:**
```python
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path='mnist_ann.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_digit(img_array):
    # img_array: numpy array of shape (28, 28), pixel values 0-255
    img = img_array.astype('float32') / 255.0
    img = img.reshape(1, 784)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    return np.argmax(probs), probs

digit, probabilities = predict_digit(your_image_array)
print(f'Predicted digit: {digit} ({probabilities[digit]*100:.1f}% confidence)')
```

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn (evaluation metrics)
- TFLite Runtime (Pi inference)

---

## 🚀 How to Run

1. Open `mnist_ann.ipynb` on [Kaggle Notebooks](https://kaggle.com)
2. No dataset setup needed — MNIST loads automatically via Keras
3. Enable GPU: Settings → Accelerator → GPU T4 (optional, trains fast on CPU too)
4. Run all cells
5. Download `mnist_ann.tflite` for Pi deployment

---

## 🔍 Key Difference from CNN

| | ANN (this project) | CNN (Cat vs Dog) |
|---|---|---|
| Input | Flattened vector (784,) | 2D image (128×128×3) |
| Layers | Dense only | Conv2D + MaxPool + Dense |
| Spatial awareness | ❌ No | ✅ Yes |
| Best for | Tabular / structured data | Image data |
| Parameters | ~567K | ~1.2M |

This project intentionally uses Dense layers only to demonstrate understanding of ANN fundamentals separate from CNNs.
