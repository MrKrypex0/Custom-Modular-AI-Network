Note: This README is a template that is not customized for this project and will be modified in the future.

# 🧠 ModularAI: Custom AI Modules with REST API & Web Crawling

![AI Network Image](./assets/banner.png)

> 🚀 Build, connect, and deploy custom AI modules like microservices — powered by PyTorch, FastAPI, and web crawling capabilities.

## 📌 Project Overview

**ModularAI** is a flexible framework for building and connecting modular AI components using **PyTorch**, designed to be deployed via a **REST API** and enhanced with **web crawling capabilities** for dynamic data ingestion.

This project allows developers to:
- Design reusable AI modules (e.g., CNNs, Transformers, Diffusion models)
- Expose them as independent services via REST endpoints
- Integrate web crawlers to fetch real-time training or inference data
- Chain modules together into complex AI pipelines

---

## 🔧 Key Features

✅ **Modular AI Architecture**  
- Each AI module is self-contained and testable  
- Designed for reusability across projects  

✅ **REST API Integration**  
- Powered by [FastAPI](https://fastapi.tiangolo.com/)   
- Enables easy communication between modules over HTTP  

✅ **Web Crawling System**  
- Built-in crawler for scraping and preprocessing web-based datasets  
- Supports structured output (JSON, CSV, etc.)  

✅ **PyTorch Native**  
- All modules are built using PyTorch for full flexibility and GPU support  

✅ **Scalable Design**  
- Can be containerized (Docker) and orchestrated (Kubernetes)

---

## 📦 Included AI Modules

| Module | Description |
|--------|-------------|
| `ConvBlock` | Feature extractor for images |
| `ResidualBlock` | Skip connection block for deep networks |
| `AttentionModule` | Soft attention mechanism |
| `DiffusionModel` | Denoising diffusion model |
| `LatentConsistencyModule` | Fast image generation |
| `VAEEncoder / VAEDecoder` | Latent space compression/decompression |
| `TransformerBlock` | Self-attention based sequence processor |
| `ClassifierHead` | Task-specific classification head |

---

## 🌐 REST API Endpoints Example

Each AI module can be exposed through its own endpoint:
```http	
POST /classify
{
"input": "image_url_or_base64",
"module": "resnet_classifier"
}

POST /generate
{
"prompt": "A cyberpunk city at night",
"model": "diffusion"
}
```

---

## 🕸️ Web Crawler Integration

The system supports dynamic data collection:

- **Crawl Mode**: Scrape websites for labeled images/text
- **Output Format**: JSON, CSV, or direct database storage
- **Preprocessing Hook**: Clean and structure data before use

Example usage:
```bash
python crawler/run.py --query "cats on skateboards" --engine google_images --limit 100
```
---
## Installation
Clone the repo:
```bash	
git clone https://github.com/yourusername/modularai.git 
cd modularai
```
Install dependencies:
```bash
pip install -r requirements.txt
```
---
## ▶️ Running the Application
Start the REST API server:
```bash
uvicorn main:app --reload
```
Visit [http://localhost:8000](http://localhost:8000) to explore the interactive API ducementation.

Run the web crawler:
```bash
python crawler/run.py --help
```
---
## 🧪 Testing AI Modules

You can test each AI component independently:
```bash
python tests/test_convblock.py
python tests/test_diffusion.py
```
---
## 🤝 Contributing
Contributions are welcome! Whether you're adding new AI modules, improving the API, or enhancing the crawler — feel free to open a PR.

To contribute:
- 1. Fork the repo
- 2. Create a feature branch (´´´bash git checkout -b feature/new_module´´´)
- 3. Commit your changes (´´´bash git commit -m 'Add new module'´´´)
- 4. Push to the branch (´´´bash git push origin feature/new_module´´´)
- 5. Open a Pull Request

---
## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) 

This project is licensed under the **Apache 2.0 License** – see the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.
---
## 📬 Questions? Feedback?

Have a question or want to suggest a feature? Feel free to open an issue or reach out to me directly.
---