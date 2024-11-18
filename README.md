# Creating-anime-characters-using-Deep-Convolutional-Generative-Adversarial-Networks-DCGANs-and-Keras

### Badges 🤖


---

# GAN for Image Generation 💡

This repository contains the implementation of a **Generative Adversarial Network (GAN)** for image generation, focusing on generating cartoon-style images. The model is built using **TensorFlow** and **Keras** frameworks, leveraging various deep learning techniques to create and evaluate generative models.

## Table of Contents 🔗

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Results](#results)
7. [Analysis](#analysis)

---

## Project Overview 🌐

The main objective of this project is to train a GAN for generating synthetic images resembling cartoon characters. The project demonstrates:
- Building **generator** and **discriminator** models.
- Training the GAN using custom datasets.
- Visualizing the real and generated data distributions.
- Optimizing the models for better accuracy and image quality.

---

## Technologies Used 🚀 

### Programming and Frameworks
- **Python 3.x**
- **TensorFlow 2.x**: For constructing and training deep learning models.
- **Keras**: For high-level API to build neural networks.

### Libraries
- **NumPy**: Array and numerical computations.
- **Matplotlib**: Data visualization.
- **Seaborn**: Enhanced statistical plotting.
- **SkillsNetwork**: Dataset preparation utilities.
- **TQDM**: Progress bars during training.

### Tools
- **Google Colab**: For cloud-based development and GPU acceleration.

---

## Setup Instructions 💻

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   If using `pip`:
   ```bash
   pip install numpy matplotlib tqdm tensorflow skillsnetwork
   ```
   Alternatively, use `mamba` for faster installations:
   ```bash
   mamba install -qy pandas numpy seaborn matplotlib scikit-learn tensorflow
   ```

3. Ensure GPU support for TensorFlow if available.

4. Download and unzip the dataset:
   ```bash
   !skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module6/cartoon_20000.zip", overwrite=True)
   ```

---

## Model Architecture 🌐

### Generator
The generator takes a latent vector and transforms it through a series of transposed convolutional layers to generate a 64x64x3 image. Each layer is followed by:
- **Batch Normalization**
- **ReLU Activation**

### Discriminator
The discriminator classifies whether an image is real or fake. It processes the input through convolutional layers with:
- **Leaky ReLU Activation**
- **Batch Normalization** (for stability)

---

## Training Process 👨‍💻

1. **Preprocessing**
   - Dataset is normalized to scale pixel values to \[-1, 1\].
   - Images are batched for efficient training.

2. **GAN Training Loop**
   - **Generator Loss**: Measures how well the generator fools the discriminator.
   - **Discriminator Loss**: Balances real vs. fake classification accuracy.
   - **Optimization**: Both networks are trained using Adam optimizers with learning rate \(0.0002\) and beta values \((0.5, 0.999)\).

3. **Epochs**: 20 epochs with a batch size of 128.

4. **Visualization**: 
   - Distribution of real vs. generated data.
   - Real and synthetic images after training.

---

## Results 💻

- **Generated Images**: Cartoon-like images created from noise vectors.
- **Accuracy**: Improved discriminator performance with balanced classification accuracy.
- **Training Metrics**: Generator and discriminator loss per epoch.

### Sample Outputs:
#### Original Dataset Samples:
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module6/cartoon_20000.zip
#### Generated Images:
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/data/generator.tar

---

## Analysis 📝

1. **Strengths**:
   - The generator successfully creates plausible images with recognizable features.
   - The discriminator efficiently identifies real and fake samples.

2. **Challenges**:
   - Requires fine-tuning hyperparameters (e.g., learning rate, noise standard deviation) for optimal results.
   - Training stability is sensitive to the initialization and batch sizes.

3. **Future Improvements**:
   - Extend the architecture to larger image resolutions.
   - Incorporate advanced loss functions (e.g., Wasserstein loss).
   - Utilize pre-trained models for transfer learning in discriminator.

---
