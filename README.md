# FashionMNIST Image Classification with PyTorch

This notebook demonstrates a complete workflow for classifying images of clothing from the FashionMNIST dataset using PyTorch. It covers both binary and multiclass classification, model training, evaluation, and visualization, with detailed explanations for each step.

## Features
- **Data Loading & Preprocessing:**
  - Downloads FashionMNIST dataset (60,000 train, 10,000 test images, 10 classes)
  - Applies normalization and tensor conversion
  - Visualizes sample images and class labels
- **Binary Classification:**
  - Filters dataset for two classes (T-shirt/top vs Ankle Boot)
  - Defines a multilayer perceptron (MLP) for binary classification
  - Trains the model and evaluates accuracy
- **Multiclass Classification:**
  - Splits data into training, validation, and test sets
  - Defines a deeper MLP for 10-class classification
  - Implements modular training and validation functions
  - Uses early stopping to prevent overfitting
  - Evaluates and visualizes predictions and accuracy
- **Visualization:**
  - Plots sample images and model predictions
  - Prints actual vs predicted labels for qualitative assessment

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies (if needed):
```bash
pip install torch torchvision matplotlib numpy
```

## How to Run
1. Open `clothesClassifier.ipynb` in Jupyter Notebook or VS Code.
2. Run each cell in order. The notebook is structured with markdown explanations before each code cell.
3. Follow the outputs and visualizations to understand each step.

## Notebook Structure
1. **Imports & Setup:** Loads libraries, sets random seed, defines transforms.
2. **Data Loading:** Downloads FashionMNIST, defines class labels, visualizes samples.
3. **Binary Classification:**
   - Data filtering, DataLoader setup
   - Model definition, training loop, evaluation
4. **Multiclass Classification:**
   - Data splitting, DataLoader setup
   - Model definition, training/validation functions
   - Early stopping, main training loop
   - Visualization and accuracy evaluation

## Key Concepts
- **DataLoader:** Efficient batching and shuffling for training/testing
- **MLP (Multilayer Perceptron):** Fully connected neural network for image classification
- **Loss Functions:**
  - `BCELoss` for binary, `CrossEntropyLoss` for multiclass
- **Optimizer:** Adam optimizer for efficient training
- **Early Stopping:** Prevents overfitting by monitoring validation loss
- **Visualization:** Matplotlib for image and result display

## Results
- The notebook prints training/validation loss per epoch and final accuracy on the test set for both binary and multiclass tasks.
- Visual and textual outputs help interpret model performance and spot misclassifications.

## Customization
- Change model architecture, batch size, or learning rate to experiment with performance.
- Try different subsets of classes or add data augmentation for further exploration.

## References
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)

---

*This notebook is designed for educational purposes and can be used as a template for other image classification tasks using PyTorch.*
