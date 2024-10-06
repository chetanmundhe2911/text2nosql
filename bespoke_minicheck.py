from minicheck.minicheck import MiniCheck
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Document statement
doc = "The PE ratio of the stock is 6.5."

# Claims to evaluate
claim_1 = "The PE ratio of the stock is 6.5."
claim_2 = "The PE ratio of the stock is 7.0."

# Using MiniCheck with the Flan-T5-Large model
scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

print(pred_label)  # Expected output: [1, 0]
print(raw_prob)    # Probabilities indicating confidence in predictions

# Alternatively, using the Bespoke-MiniCheck-7B model
scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False, cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

print(pred_label)  # Expected output: [1, 0]
print(raw_prob)    # Probabilities indicating confidence in predictions
