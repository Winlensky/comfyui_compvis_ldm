# ComfyUI Custom Nodes for Latent Diffusion (CompVis F8 Large)
<img width="1583" height="852" alt="image" src="https://github.com/user-attachments/assets/b76c6c5e-e0f9-49ca-a78a-93b0e882f990" />

A set of custom nodes for **ComfyUI** designed to natively initialize and run the historic original **Latent Diffusion Model (LDM) F8 Large** architecture from CompVis. 

Unlike newer models (like Stable Diffusion 1.5), this specific pipeline relies on a heavy **BERT-based** text encoder (`dim: 1280`, `vocab_size: 30522`) and an aggressive `f8` latent autoencoder. This repository enables standalone support for this unique model without relying on external `.yaml` configuration blueprints or old, deprecated codebases.

## 🚀 Key Features
* **Native LDM F8 Large Support:** Full execution pipeline for text-to-image (txt2img) generation using the original architecture weights.
* **Lightweight Dependencies:** No bloatware or heavy framework wrappers. Built on top of core packages (`torch`, `safetensors`, `transformers`) already shipped with standard ComfyUI setup.
* **Zero-Config Loading:** The architecture is handled directly within the custom nodes backend, preventing configuration errors common with standard checkers.

## 📦 Installation

### For Windows and Linux
1. Navigate to your ComfyUI custom nodes directory inside your terminal or PowerShell.
2. Clone this repository:
   ```powershell
   git clone https://github.com/Winlensky/comfyui_compvis_ldm.git
   ```
3. Install requirements (Most likely not needed):
```
cd comfyui_compvis_ldm
pip install -r requirements.txt
```
5.  Restart ComfyUI. The nodes will be registered automatically using your embedded Python environment.

## 🗂️ Model Downloads

* **Download Checkpoints:** [Latent Diffusion F8 Large on Hugging Face](https://huggingface.co/Winlensky/latent-diffusion-f8-large-safetensors/tree/main)
* **Available Flavors:** 
  * `latent-diffusion-f8-large-fp32.safetensors` (Original precision, ~6GB)
  * `latent-diffusion-f8-large-bf16.safetensors` (Optimized precision, ~3GB)
  * `latent-diffusion-f8-large-fp16.safetensors` (Optimized precision, ~3GB)
  * `latent-diffusion-f8-large-jack000-finetuned-fp16.safetensors` (Finetuned version by jack000 for better structural composition)

### 📂 Where to place the files:
Place the downloaded `.safetensors` files strictly into your ComfyUI models directory under the dedicated LDM folder: `ComfyUI\models\ldm`

## 🛠️ Usage & Node Workflows

* **Resolution Notice:** The LDM f8-large model was explicitly trained on downscaled, tight matrix resolutions. Ensure your target generation canvas is constrained (e.g., `256x256`).

## 📜 Credits & License
* Core architecture by **CompVis (LMU Munich)**.
* Finetuned variations by **jack000**.
* This project is distributed under the **MIT License**.
