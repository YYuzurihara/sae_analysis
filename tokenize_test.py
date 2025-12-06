from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", device="cpu")

text = "Hello, world!"
tokens = model.to_str_tokens(text, prepend_bos=True)
print(f"Tokens: {tokens}")
# INSERT_YOUR_CODE
# token idのリストに直して出力してください
token_ids = model.to_tokens(text, prepend_bos=True).tolist()
print(f"Token IDs: {token_ids}")