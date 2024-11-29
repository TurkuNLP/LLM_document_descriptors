import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch._dynamo

if __name__ == "__main__":

    torch._dynamo.config.suppress_errors = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device {device}.")

    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.bfloat16,
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_quant_storage=torch.bfloat16,
    #)

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    #tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    #if tokenizer.eos_token_id is None:
    #    tokenizer.eos_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map='auto', # device_map='auto' to automatically split to all GPUs
                                                 torch_dtype=torch.bfloat16, # Use bfloat16 precision, which halves memory demand without impacting performance
                                                )
                                                 #attn_implementation="flash_attention_2") # Flash attention 2 for better performance

    print('Parameters on meta device:')
    for n, p in model.named_parameters():
        if p.device.type == "meta":
            print(f"{n} is on meta!")

    print()

    # Optimise model performance with static kv-cache: https://huggingface.co/docs/transformers/v4.46.0/llm_optims
    # Only works on Llama-models, for now.
    #model.generation_config.cache_implementation = "static"
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    messages = [
        {
            "role": "system",
            "content": "You are an expert in multiple tasks. You only responed truthfully and briefly.",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
     ]
    
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    print('Tokenized chat')
    print(tokenized_chat)
    #attention_mask = (tokenized_chat != tokenizer.pad_token_id)
    with torch.no_grad():
        outputs = model.generate(tokenized_chat, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        with open('../results/test_generation.txt', 'w') as f:
            #f.write(tokenizer.decode(tokenized_chat))
            #f.write('\n')
            f.write(tokenizer.decode(outputs[0]))
