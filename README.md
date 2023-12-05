# GPT2-125M-replication
Implementation of GPT-2 small model with 125M parameters along with changes in architecture such as Rotary position embeddings and grouped query attention.

## Implementation of GPT-2 small
## Transformer Architectural Change
### >> Rotary Embedding integration
### >> Grouped query attention integration
## Single GPU, DDP, FSDP training loops

## References
Attention research paper - https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf 

GPT research paper - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

GPT2 research paper - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Github used - https://github.com/karpathy/nanoGPT/blob/master/model.py#L6

Huggingface resources used - https://huggingface.co/gpt2, https://huggingface.co/transformers/v3.0.2/_modules/transformers/tokenization_gpt2.html#GPT2Tokenizer, https://discuss.huggingface.co/t/how-to-decode-gpt2/16160

Dataset -- https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

DDP - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

FSDP - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp

Rotary Embeddings Github - https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

GQA research paper - https://arxiv.org/pdf/2305.13245v2.pdf

GQA Github - https://github.com/knotgrass/attention/blob/main/attn/attention.py
