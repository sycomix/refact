import json

import torch

from hf.modeling_gpt_refact import GPTRefactForCausalLM
from hf.configuration_gpt_refact import GPTRefactConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as th

from refact_models import RefactModel


def load_model(model):
    d = torch.load(
        "/Users/kirillstarkov/WORK/storage/202306-refact2b-mqa-alibi-lion-1040B-s06-toks-cont/000164500/mp_rank_00_model_states.pt",
        map_location=torch.device('cpu'))["module"]
    model.transformer.wte.weight.data[:] = d.pop("emb.layer.weight").to(th.float32)
    model.ln_f.weight.data = d.pop("final.ln.weight").to(th.float32)
    model.lm_head.weight.data = d.pop("final.unemb.layer.weight").to(th.float32)
    for i, layer in enumerate(model.transformer.h):
        prefix = f'layers.{i+1:03}'
        model.transformer.h[i].ln_1.weight.data = d.pop(f'{prefix}.ln_a.weight').to(th.float32)
        model.transformer.h[i].ln_2.weight.data = d.pop(f'{prefix}.ln_m.weight').to(th.float32)

        model.transformer.h[i].mlp.linear_1.weight.data = d.pop(f'{prefix}.pw.linear_1.weight').to(th.float32)
        model.transformer.h[i].mlp.c_proj.weight.data = d.pop(f'{prefix}.pw.linear_2.weight').to(th.float32)
        model.transformer.h[i].mlp.linear_3.weight.data = d.pop(f'{prefix}.pw.linear_3.weight').to(th.float32)

        model.transformer.h[i].attn.q.weight.data = d.pop(f'{prefix}.mha.q.weight').to(th.float32)
        model.transformer.h[i].attn.k.weight.data = d.pop(f'{prefix}.mha.k.weight').to(th.float32)
        model.transformer.h[i].attn.v.weight.data = d.pop(f'{prefix}.mha.v.weight').to(th.float32)
        model.transformer.h[i].attn.c_proj.weight.data = d.pop(f'{prefix}.mha.out.weight').to(th.float32)

    return model



if __name__ == "__main__":
    with open("/Users/kirillstarkov/WORK/storage/202306-refact2b-mqa-alibi-lion-1040B-s06-toks-cont/000164500/model-hps.json", "r") as f:
        cfg = json.load(f)
    config = GPTRefactConfig(
        vocab_size=49216,
        n_embd=cfg["E"],
        n_layer=cfg["L"],
        n_head=cfg["attn_heads"],
        do_sample=True
    )
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

    # or_model = RefactModel.from_pretrained("/Users/kirillstarkov/WORK/storage/202306-refact2b-mqa-alibi-lion-1040B-s06-toks-cont/000164500", "cpu")
    # mask = torch.ones((10, 10), dtype=torch.bool, device='cpu')
    # mask = torch.triu(mask, 1)
    # qwe = or_model(th.ones(1, 10, dtype=th.int64), mask, use_cache=True)
    model = load_model(GPTRefactForCausalLM(config))
    # model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")
    # model.eval()
    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt")

    outputs = model.generate(inputs, use_cache=True)
    print(tokenizer.decode(outputs[0]))
    i = 0

