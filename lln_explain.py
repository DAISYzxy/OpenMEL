import fire
from typing import List, Dict, Any, Optional
from llama import Dialog, Llama
import json
from tqdm import tqdm
import os
import torch  # Import torch here
import torch.distributed as dist


# Function to ensure we can write to a directory
def check_write_permission(dir_path):
    if not os.access(dir_path, os.W_OK):
        raise PermissionError(f"No write permission to directory: {dir_path}")

# Set environment variables for distributed processing if needed
if dist.is_available() and torch.cuda.is_available():
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP of the master node
    os.environ['MASTER_PORT'] = '12356'  # some unused port number
    os.environ['WORLD_SIZE'] = '1'  # total number of processes
    os.environ['RANK'] = '0'  # rank of this process

    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://')

# Read Json file
def read_json_file(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Initialize the global model
def initialize_model(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int
):
    global generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=1  # Specify model parallel parameter
    )

def main(
    origin_text: str,
    mention_name: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    dialogs: List[Dialog] = [
        [
            {
                "role": "system",
                "content": "You will be given a text and a specific mention. We will further link the mention to an entity in the existing knowledge base. Please provide more information about this mention which will help our further linking. Please give the information following the example. Given text: Parizeau at a 1981 conference at Laval University. mention: Parizeau. Information you should generated: Jacques Parizeau is the Canadian politician. Second example. Given text: Girolamo Panzetta in September 2007. mention: Girolamo Panzetta. The information you should generated: Girolamo Panzetta is the Italian actor. DO NOT give any other additional sentence except the information.",
            },
            {
                "role": "user",
                "content": f"Given text: {origin_text}, Mention: {mention_name}",
            },
        ]
    ]
    
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print(
        f"> {results[0]['generation']['role'].capitalize()}: {results[0]['generation']['content']}"
    )

    return results[0]['generation']['content']

def run_with_defaults(origin_text, mentions):
    # Check if mentions is a list and extract first element
    if isinstance(mentions, list) and len(mentions) > 0:
        mention_name = mentions[0]
    else:
        mention_name = mentions
    
    temperature = 0.7
    top_p = 0.9
    max_seq_len = 512
    max_batch_size = 4
    max_gen_len = None

    result = main(
        origin_text=origin_text,
        mention_name=mention_name,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_gen_len=max_gen_len,
    )
    return result

if __name__ == "__main__":
    ckpt_dir = "Meta-Llama-3-8B-Instruct/"
    tokenizer_path = "Meta-Llama-3-8B-Instruct/tokenizer.model"
    max_seq_len = 512
    max_batch_size = 4

    # Initialize the model once
    initialize_model(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )
    
    data = read_json_file("") # fill in the corresponding file path
    processed_data = []
    for instance in tqdm(data):
        parsed_instance = dict()
        parsed_instance["id"] = instance["id"]
        parsed_instance["sentence"] = instance["sentence"]
        parsed_instance["mentions"] = instance["mentions"]
        parsed_instance["entities"] = instance["entities"]
        parsed_instance["answer"] = instance["answer"]
        parsed_instance["imgPath"] = instance["imgPath"]
        
        try:
            parsed_instance["explain"] = run_with_defaults(parsed_instance["sentence"], parsed_instance["mentions"])
        except Exception as e:
            print(f"Error processing instance {instance['id']}: {e}")
            parsed_instance["explain"] = None
        
        print(parsed_instance["explain"])
        processed_data.append(parsed_instance)

    check_write_permission('') # fill in the corresponding folder path
    with open('', 'w', encoding='utf-8') as f: # fill in the saved file path
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
