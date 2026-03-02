# AIG230 – Assignment 6 

Neural Language Model using **PyTorch RNN** on the **NLTK Brown corpus (`news`)**.

## Implementation Summary

The notebook implements all required Part B components:

1. **B1: Numericalization + training examples**
	- Lowercasing and punctuation-only token removal
	- Adds `<bos>` and `<eos>` to each sentence
	- Builds vocabulary from train split only with `min_freq=2`
	- Maps OOV words to `<unk>`
	- Creates next-token prediction pairs from token streams:
	  - input: `x = stream[i:i+T]`
	  - target: `y = stream[i+1:i+T+1]`

2. **B2: RNN language model**
	- `nn.Embedding(vocab_size, emb_dim)`
	- `nn.RNN(emb_dim, hid_dim, batch_first=True)`
	- `nn.Linear(hid_dim, vocab_size)`

3. **B3: Training + validation**
	- Loss: `nn.CrossEntropyLoss()`
	- Optimizer: `torch.optim.Adam(..., lr=1e-3)`
	- Gradient clipping: `clip_grad_norm_` with `grad_clip=1.0`
	- Validation metric: perplexity

4. **B4: Test perplexity + generation**
	- Evaluates test perplexity after training
	- Generates text token-by-token starting from `<bos>` with temperature sampling

## Configuration

- Category: `news`
- Split ratio: `80% / 10% / 10%` (train/val/test)
- `min_freq=2`, `seq_len=30`, `batch_size=32`
- `emb_dim=128`, `hid_dim=256`, `num_layers=1`, `dropout=0.0`
- `epochs=5`, `lr=1e-3`

## Dataset & Vocabulary Results

- Raw sentences: **4623**
- Train: **3698 sentences**, **77,918 tokens**
- Validation: **462 sentences**, **10,264 tokens**
- Test: **463 sentences**, **9,656 tokens**
- Vocabulary size: **5353** (`UNK id = 2`)
- Train examples (stream dataset): **77,888**

## Model & Training Results

- Device used: **CPU**
- Model parameters: **2,159,721**
- Batch shape check:
  - `x_batch`: `torch.Size([32, 30])`
  - `y_batch`: `torch.Size([32, 30])`

### Epoch Metrics

- Epoch 1: train loss **3.6107**, val ppl **431.68**
- Epoch 2: train loss **1.2977**, val ppl **1730.75**
- Epoch 3: train loss **0.6994**, val ppl **4389.29**
- Epoch 4: train loss **0.5123**, val ppl **8420.00**
- Epoch 5: train loss **0.4329**, val ppl **12981.14**

## Final Test Result

- Test perplexity: **12166.391876923895**

## Sample Generated Text

1. `<bos> richard m. <unk> below the couple of professional posts at the united city <eos>`
	- Comment: Produces a short news-like fragment with named entities, but still includes `<unk>` and ends early.

2. `<bos> in the third frank robinson hammered a long home run deep into the corner of the <unk> in church <unk> and the christian <unk> are there be said <unk> look <unk> can be a serious health <unk> <eos>`
	- Comment: Longer sequence with sports/news structure, but coherence drops later and unknown tokens appear frequently.

3. `<bos> <unk> <eos>`
	- Comment: Very short collapsed output, suggesting unstable next-token predictions for some sampling paths.

## Notes

- The notebook includes a training loss plot (`matplotlib`).
- Validation and test perplexity are very high despite decreasing train loss, indicating likely overfitting / limited generalization with the current setup.
