# Multiclass Image Classification using CNN

This repository contains code for a custom-built Convolutional Neural Network (CNN) designed from scratch to predict multiclass images. The model achieves high accuracy in classifying images into various categories, including sea, mountains, forests, buildings, glaciers, and streets.

## Overview

In this project, we develop a CNN architecture tailored specifically for multiclass image classification tasks. The CNN is trained on a diverse dataset comprising images of different landscapes, such as sea, mountains, forests, buildings, glaciers, and streets. The goal is to accurately predict the class label associated with each image.

## Key Features

- Custom CNN architecture designed from scratch for multiclass image classification.
- Dataset containing images of diverse landscapes for training and evaluation.
- Training pipeline to efficiently train the CNN model.
- Evaluation metrics to assess the performance of the model, including accuracy and loss.
- Prediction functionality to classify new images using the trained model.

## Requirements

- Python (>=3.6)
- TensorFlow
- NumPy
- Matplotlib
- sklearn

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/multiclass-image-classification.git
    cd multiclass-image-classification
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your dataset:**

   - Organize your dataset into appropriate directories, each representing a different class (e.g., sea, mountains, forests, buildings, glaciers, streets).
   - Ensure that images are properly labeled and formatted.

4. **Train the CNN model:**

    ```bash
    python train.py --dataset_path /path/to/dataset --epochs 50 --batch_size 32
    ```

   Adjust the parameters (e.g., number of epochs, batch size) as needed.

5. **Evaluate the trained model:**

    ```bash
    python evaluate.py --model_path /path/to/saved_model --dataset_path /path/to/test_dataset
    ```

6. **Make predictions:**

    ```bash
    python predict.py --model_path /path/to/saved_model --image_path /path/to/image
    ```

   Replace `/path/to/...` with the appropriate file paths.

## Results

Provide insights into the performance of the model, including accuracy achieved on the test dataset and any notable observations or challenges encountered during training and evaluation.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request. All contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Acknowledge any resources, libraries, or tutorials used in the development of this project.

## Demo

---
