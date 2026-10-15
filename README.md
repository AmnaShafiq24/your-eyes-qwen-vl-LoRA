# Your Eyes: PKR Currency + Fruit/Vegetable Recognition (Qwen2.5-VL-3B LoRA)

A LoRA adapter for [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) that looks at a photo and answers with a short, speech-friendly name, for example `1000 rupee note` or `banana`. It was built for **Your Eyes**, an Android app that helps blind and visually impaired users identify everyday items.

> **Important:** this is a research/prototype model. It can be wrong, and a wrong banknote denomination is a serious mistake for someone who cannot double-check visually. Do not use it as the only source of truth for money.

## What it recognises

| Category | Classes |
|---|---|
| Pakistani rupee notes | 10, 20, 100, 500, 1000, 5000 |
| Fruits & vegetables | apple, banana, orange, mango, grape, pear, pineapple, watermelon, kiwi, lemon, pomegranate, strawberry, peach, plum, potato, tomato, onion, carrot, cucumber, pepper, cabbage, cauliflower, corn, garlic, ginger, eggplant, beetroot |

Prompt used for training and inference:

```
What is shown in this image? Answer with just the item name.
```

## Repository layout

```
.
├── adapter/                      # LoRA adapter + tokenizer/processor files (weights via Git LFS)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer / processor files
├── finetuning.ipynb   # Colab training notebook
├── README.md
└── .gitignore
```

## Training summary

| | |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct (Unsloth mirror, 16-bit weights) |
| Method | LoRA (r = 16) via [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer` |
| Data | PKR currency dataset (Kaggle) + Fruits-360 (Kaggle) |
| Training images | 5,679 (about 6.3k total; 90/10 train/validation split) |
| Image preprocessing | Longest side resized to **448 px** |
| Batch size / epochs | 16 / 2 (710 steps) |
| Learning rate | 2e-4, linear schedule, 8-bit AdamW |
| Hardware / time | 1x A100, about 20 minutes |

Resizing images before training cut training time from a projected ~48 hours to about 20 minutes. Full-resolution photos produce thousands of visual tokens per image.

## Results

Validation accuracy on 300 random held-out images (loose "contains" match):

| Category | Accuracy |
|---|---|
| Currency | 177 / 180 (98.3%) |
| Fruit / vegetable | 118 / 120 (98.3%) |
| **Overall** | **295 / 300 (98.3%)** |



## Usage

**Important:** resize the image so its longest side is at most 448 px before sending it to the model, exactly as in training.

### With Transformers + PEFT

```python
import torch
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = "What is shown in this image? Answer with just the item name."

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "adapter")      # path to this repo's adapter/ folder
processor = AutoProcessor.from_pretrained(BASE)

image = ImageOps.exif_transpose(Image.open("photo.jpg")).convert("RGB")
image.thumbnail((448, 448))

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
```

### With Unsloth

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained("adapter", load_in_4bit=False)
FastVisionModel.for_inference(model)
# then build the prompt as in the notebook's ask() function
```

A GPU is needed for usable latency. Serve it on a small GPU host (for example vLLM).

## Reproducing the training

Open `finetuning.ipynb` in Google Colab with an A100 runtime, upload your `kaggle.json` when prompted, and run the cells top to bottom.

## Limitations

- Validation images come from the same sources as the training images. Fruits-360 contains near-duplicate frames, so the score is optimistic. Accuracy on real phone photos (lighting, folded notes, hands, clutter) will be lower. Test on your own photos.
- It always gives an answer, even for objects outside its classes, and it gives no confidence score.
- Never use this as the only check for banknote denominations. Have the app ask for a retake when unsure.
- Only the listed classes were trained.

## Credits and licenses

- Base model: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) by the Qwen team. Check its license on the model page before using or redistributing.
- Fine-tuning: [Unsloth](https://github.com/unslothai/unsloth).
- Data: [Pakistani Currency Dataset](https://www.kaggle.com/datasets/muhammadammarabid/pakistanicurrencydataset) and [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits). Check each dataset's license and cite the authors.


## Author

Amna Shafiq / amnashafiq1024@gmail.com

