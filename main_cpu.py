import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Tokenize the input text
input_ids = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt").input_ids




# We will generate one token at a time and print it
output_ids = input_ids
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for _ in range(200):  # Set the maximum length of the generation
        outputs = model(output_ids)
        next_token_logits = outputs.logits[:, -1, :]  # Get the last token's logits
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Choose the most probable next token
        output_ids = torch.cat((output_ids, next_token), dim=1)  # Append the new token to the output ids

        # Decode and print the token
        word = tokenizer.decode(next_token.squeeze().tolist())
        print(word, end='')

        # Stop if the end of text token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
