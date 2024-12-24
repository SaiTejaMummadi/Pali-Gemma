import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    #Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding)