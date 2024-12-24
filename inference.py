from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def main(
        model_path: str = None,
        prompt: str = None, 
        image_file_path: str = None,
        max_tokens_to_generate: int = 100,
        temperature: float = 0.8,
        top_p: float=0.9,
        do_sample: bool = False,
        only_cpu: bool = False,
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    
    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer =load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running Inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

   

if __name__ == "__main__":
    fire.Fire(main)