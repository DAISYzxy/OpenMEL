import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import json
import os

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)

model = model.to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

with open("", "r") as f: # fill in the corresponding file path
    mention = json.load(f)

captions = dict()
folder_path = '' # fill in the corresponding folder path

for idx in tqdm(range(len(mention))):
    item = mention[idx]
    mention_img = item["imgPath"]
    raw_text = item["sentence"]
    mention_name = item["mentions"]
    if not mention_img.endswith(".jpg"):
        mention_img = mention_img.rsplit('.', 1)[0] + ".jpg"
    if len(mention_img) > 1:
        filename = mention_img
        img_path = os.path.join(folder_path, filename)
        
        # Check if the image file exists
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            question = f"The image is about {mention_name}. Also give you related context that {raw_text}. Give the possible occupation of the person in the figure WITHOUT explanations. Just generate in one sentence."
            msgs = [{'role': 'user', 'content': question}]
            
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True, # if sampling=False, beam_search will be used by default
                temperature=0.7,
                # system_prompt='' # pass system_prompt if needed
            )
            # print(res)
            captions[filename] = res
        else:
            print(f"File not found: {img_path}")


with open("", 'w', encoding='utf-8') as f: # fill in the saved file path
    json.dump(captions, f, ensure_ascii=False, indent=4)
