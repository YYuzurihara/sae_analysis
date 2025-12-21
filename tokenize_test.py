from transformer_lens import HookedTransformer
from prompt_hanoi import get_answer, POS_TO_START_SOLVE

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", device="cpu")

text = get_answer()
tokens = model.to_str_tokens(text, prepend_bos=True)
print(f"length of tokens: {len(tokens)}")
print(f"Tokens: {tokens[POS_TO_START_SOLVE:]}")

print("prompt:")
print(''.join(tokens[:POS_TO_START_SOLVE]), end="\n\n")

print("target:")
print(''.join(tokens[POS_TO_START_SOLVE:]), end="\n\n")

# # token_ids = model.to_tokens(text, prepend_bos=True).tolist()
# print(f"length of token_ids: {len(token_ids[0])}")
# print(f"Token IDs: {token_ids[0][:POS_TO_START_SOLVE]}")