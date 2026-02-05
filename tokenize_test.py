from prompt_addition import get_answer
from plot_prob import load_model
from model_config import llama_scope_lxr_32x, llama_scope_r1_distill
import torch

# Load tokenizers
print("Loading Llama tokenizer...")
llama_config = llama_scope_lxr_32x("cpu", 0)
llama_model, _ = load_model(llama_config, sae=None, skip_hf_model=True)

print("Loading DeepSeek-R1-Distill-Llama tokenizer...")
deepseek_config = llama_scope_r1_distill("cpu", 0)
deepseek_model, _ = load_model(deepseek_config, sae=None, skip_hf_model=False)

# Test all combinations of op1 and op2 from 0 to 1000
# print("\nTesting tokenization for op1, op2 in range [0, 1000]...\n")

# mismatch_count = 0
# total_count = 0
# token_length_list = []
# for op1 in range(1001):
#     for op2 in range(1001):
#         text, _ = get_answer(op1, op2)
        
#         # Get string tokens from both models (text segmentation)
#         llama_str_tokens = llama_model.to_str_tokens(text, prepend_bos=True)
#         deepseek_str_tokens = deepseek_model.to_str_tokens(text, prepend_bos=True)
        
#         total_count += 1
#         token_length_list.append(len(llama_str_tokens))
        
#         # Check if tokenization (text segmentation) matches
#         if llama_str_tokens != deepseek_str_tokens:
#             mismatch_count += 1
#             if mismatch_count <= 10:  # Print first 10 mismatches
#                 print(f"MISMATCH at op1={op1}, op2={op2}")
#                 print(f"  Text: {text}")
#                 print(f"  Llama tokens:    {llama_str_tokens}")
#                 print(f"  DeepSeek tokens: {deepseek_str_tokens}")
#                 print()

# print(f"\nResults:")
# print(f"Total combinations tested: {total_count}")
# print(f"Mismatches found: {mismatch_count}")
# print(f"Match rate: {(total_count - mismatch_count) / total_count * 100:.2f}%")

# if mismatch_count == 0:
#     print("\n✓ All tokenizations match!")
#     print(f"Token length distribution: {set(token_length_list)}")
# else:
#     print(f"\n✗ Found {mismatch_count} tokenization mismatches")
#     print(f"Token length distribution: {token_length_list}")

while True:
    op1, op2 = input("Enter op1 and op2: ").split()
    text, _ = get_answer(int(op1), int(op2))
    llama_str_tokens = llama_model.to_str_tokens(text, prepend_bos=True)
    deepseek_str_tokens = deepseek_model.to_str_tokens(text, prepend_bos=True)
    print(f"Llama tokens: {llama_str_tokens}")
    print(f"DeepSeek tokens: {deepseek_str_tokens}")