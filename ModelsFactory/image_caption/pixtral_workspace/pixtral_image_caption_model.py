from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

model_id = "mistral-community/pixtral-12b"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda")

url_dog = "https://picsum.photos/id/237/200/300"
url_mountain = "https://picsum.photos/seed/picsum/200/300"

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can this animal"},
        {"type": "image"},
        {"type": "text", "content": "live here?"},
        {"type": "image"}
      ]
    }
]

prompt = processor.apply_chat_template(chat)
inputs = processor(text=prompt, images=[url_dog, url_mountain], return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]