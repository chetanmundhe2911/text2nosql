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

#Using deberta V3
# Using MiniCheck with DeBERTa-v3-large model
scorer = MiniCheck(model_name='deberta-v3-large', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc]*4, claims=[claim_1, claim_2, claim_3, claim_4])

print(pred_label)  # Expected output: [1, 1, 1, 0]
print(raw_prob)    # Probabilities indicating confidence in predictions



# Another example:

from minicheck.minicheck import MiniCheck
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Document statement
doc = "The company has a PE ratio of 6.5, an ROE of 15%, and an ROCE of 12%."

# Claims to evaluate
claim_1 = "The PE ratio is 6.5."
claim_2 = "The ROE is 15%."
claim_3 = "The ROCE is 12%."
claim_4 = "The ROE is 20%."

# Using MiniCheck with the Flan-T5-Large model
scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc]*4, claims=[claim_1, claim_2, claim_3, claim_4])

print(pred_label)  # Expected output: [1, 1, 1, 0]
print(raw_prob)    # Probabilities indicating confidence in predictions

# Alternatively, using the Bespoke-MiniCheck-7B model
scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False, cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc]*4, claims=[claim_1, claim_2, claim_3, claim_4])

print(pred_label)  # Expected output: [1, 1, 1, 0]
print(raw_prob)    # Probabilities indicating confidence in predictions
