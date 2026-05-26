import gradio as gr
from aura_sr import AuraSR
from PIL import Image
import torch

# MPS対応確認
print('PyTorch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())

model = None
def load_model():
    global model
    if model is None:
        model = AuraSR.from_pretrained('fal/AuraSR-v2', device='mps' if torch.backends.mps.is_available() else 'cpu')
    return model

def upscale(image):
    if image is None:
        return None
    load_model()
    # メモリ節約のためサイズ制限
    max_dim = 1024
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    upscaled = model.upscale_4x_overlapped(image)
    return upscaled

iface = gr.Interface(
    fn=upscale,
    inputs=gr.Image(type="pil", label="アップスケールしたい画像 (1024px以下推奨)"),
    outputs=gr.Image(type="pil", label="4x アップスケール後画像"),
    title="AuraSR-v2 - Mac対応 4x Super Resolution",
    description="Apple Silicon Mac (M1/M2/M3/M4) で簡単に使えるGAN超解像ツール\n\n入力画像は小さめがおすすめ (OOM防止)",
    examples=None,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=False) # share=Trueにすれば公開可能
