# 🧠 Modular AI Components in PyTorch

This document introduces key **AI modules** that can be implemented in **Pytorch**. These modules are building blocks for creating complex, composable AI architectures - from vision models to diffusion pipelines and beyond.

Each module is designed to be reusable, and combinable with others to form more advanced AI systems.

---

## 🔹 1. `ConvBlock` – Convolutional Feature Extractor

### 📌 Description:
A basic convolutional block used for feature extraction in images.

### ✅ Purpose:
To extract spatial features from images inputs using convolutional layers.

### 🧪 When to Use:
In CNNs, object detection, segmentation, or as part of larger architectures like UNet or ResNet.

### 🧩 Example Code:
```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### 📚 Resources:

- [Pytorch Convolutional Layers](https://docs.pytorch.org/docs/stable/nn.html#convolution-layers)

- [Deep Learning for Computer Vision](https://cs231n.github.io/)

---

## 🔹 2. `ResidualBlock` – Identity-Preserving Skip Connection Block


### 📌 Description:
Implements skip connections to allow gradients to flow through deep networks without vanishing.

### ✅ Purpose:
To improve training stability in deep CNNs by preserving identity information.

### 🧪 When to Use:
In ResNets or any deep CNN architecture.

### 🧩 Example Code:
```python
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.con1 = ConvBlovk(channels, channels)
        self.con2 = nn.conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.con1(x)
        x = self.con2(out)
        x += residual
        return F.relu(out) 
```

### 📚 Resources:

- [Original ResNet Paper](https://arxiv.org/abs/1512.03385)

---
## 🔹 3. `AttentionBlock` – Soft Attention Mechanism

### 📌 Description:
Applies attention to focus on important parts of input sequences or images.

### ✅ Purpose:
To dynamically weight inputs based on relevance (e.g., in transformers or CNNs).

### 🧪 When to Use:
In NLP tasks, image captioning, or generative models.

### 🧩 Example Code:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)))
        return torch.matmul(attention_weights, V)
```

### 📚 Resources:

- [Attention is all you need](https://arxiv.org/abs/1706.03762)

---
## 🔹 4. `DiffusionModel` – Denoising Diffusion Probalistic Model

### 📌 Description:
Generates data by iteratively denoising latent representations.

### ✅ Purpose:
High-quality image generation and manipulation.

### 🧪 When to Use:
Text-to-image synthesis, inpainting, style transfer.

### 🧩 Example Code:
```python
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, unet, beta_schedule):
        super(DiffisionModel, self).__init__()
        self.unet = unet
        self.register_buffer('beta', beta_schedule)
        self.alpha = 1 - self.beta

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        pred_noise = self.unet(noisy_x, t)
        return noise, pred_noise
```

### 📚 Resources:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Hugging Face Diffusers Docs](https://huggingface.co/docs/diffusers/index)

---
## 🔹 5. `LatentConsistencyModule` – Fast Generative Inference via Consistency

### 📌 Description:
Accelerates diffusion model inference to 1–4 steps using consistency learning.

### ✅ Purpose:
Fast image generation while maintaining quality.

### 🧪 When to Use:
Real-time applications, mobile deployment, or UI interaction.

### 🧩 Example Code:
```python
import torch
import torch.nn as nn

class LatentConsistencyModule(nn.Module):
    def __init__(self, prettrained_diffusion_model):
        super(LatentConsistencyModule, self).__init__()
        self.model = pretrained_diffusion_model
        # Freeze weights if distilling
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x, t):
        # Apply consistency mapping
        output = self.model(latent, step)
        return output
```

### 📚 Resources:
- [LCM: Latent Consistency Distillation](https://arxiv.org/abs/2310.04376)
- [Microsoft Github Repository (LCM)](https://github.com/microsoft/LCM)

---
## 🔹 6. `VAEEncoder / VAEDecoder` – Variational Autoencoder

###  📌 Description:
Encodes input into a ltent distribution and decodes it back.

### ✅ Purpose:
Dimensionality reduction, latent representation learning, anomaly detection.

### 🧪 When to Use:
In autocoders, GANs, or diffusion models.

### 🧩 Example Code:
```python
import torch 
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
```

### 📚 Resources:
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

---
## 🔹 7. `TransformerBlock` – Self-Attention Based Sequence Module

### 📌 Description:
Implements a single transformer layer with attention and feedforward components.

### ✅ Purpose:
Process sequential data like text, audio, or time series.

### 🧪 When to Use:
Language modeling, machine translation, video processing.

### 🧩 Example Code:
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x
```

### 📚 Resources:
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
---
## 🔹 8. `ClassifierHead` – Task-Specific Output Head

### 📌 Description:
A classification head that maps latent features to class labels.

### ✅ Purpose:
To perform classification on top of feature encoders.

### 🧪 When to Use:
Supervised fine-tuning, multi-task learning.

#### 🧩 Example Code:
```python
import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead, self).__init__()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.head(x)
```
---
## 🤖 Building a Network of Communicating AI Modules
You can compose these modules into a pipeline or network where outputs from one module feed into another.

### 🧱 Example: Image Generation Pipeline
```python
import torch.nn as nn

class ImageGenerationPipeline(nn.Module):
    def __init__(self):
        super(ImageGenerationPipeline, self).__init__()
        self.encoder = VAEEncoder(latent_dim=128)
        self.diffusion = DiffusionModel(unet=UNet(), beta_schedule=get_beta_schedule())
        self.decoder = VAEDecoder(latent_dim=128)

    def forward(self, x, t):
        z, _, _ = self.encoder(x)
        noisy_z = self.diffusion(z, t)
        img = self.decoder(noisy_z)
        return img
```       

### 🔄 Communication Between Modules
```python
def run_pipeline(image_input, timestep):
    z = encoder(image_input)
    refined_z = diffusion(z, timestep)
    output_image = decoder(refined_z)
    return output_image
```

### 📚 Additional Resources

| TOPIC    | RESOURCES |
|----------|-----------|
| PyTorch Official Docs    | [https://pytorch.org/docs](https://docs.pytorch.org/docs/stable/index.html)    |
| HuggingFace Transformers    | [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/index)    |
| Diffusers Library | [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers/index)   |
| Papers with Code | [https://paperswithcode.com](https://paperswithcode.com/) |
| PyTorch Lightning | [https://www.pytorchlightning.ai](https://lightning.ai/) |

### 🧩 Bonus: Modular Design Principles

- **Single Responsibility**: Each module does one thing well.
- **Composability**: Combine modules to build complex systems.
- **Reusability**: Reuse across different projects/tasks.
- **Testability**: Unit-test each module independently.

---

