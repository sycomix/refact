huggingface_mini_db = {
    "starcoder/15b/base": {
        "backend": "autogptq",
        "model_path": "TheBloke/starcoder-GPTQ",
        "diff_scratchpad_class": "refact_scratchpads:ScratchpadHuggingface",
        "chat_scratchpad_class": None,
        "model_class_kwargs": {},
        "filter_caps": ["completion"],
    },
    "starcoder/15b/plus": {
        "backend": "autogptq",
        "model_path": "TheBloke/starcoderplus-GPTQ",
        "diff_scratchpad_class": "refact_scratchpads:ScratchpadHuggingface",
        "chat_scratchpad_class": None,
        "model_class_kwargs": {},
        "filter_caps": ["completion"],
    },
    "starchat/15b/beta": {
        "backend": "autogptq",
        "model_path": "TheBloke/starchat-beta-GPTQ",
        "diff_scratchpad_class": None,
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceStarChat",
        "model_class_kwargs": {},
        "filter_caps": ["starchat"],
    },
    "wizardcoder/15b": {
        "backend": "autogptq",
        "model_path": "TheBloke/WizardCoder-15B-1.0-GPTQ",
        "diff_scratchpad_class": "refact_scratchpads:ScratchpadHuggingface",
        "chat_scratchpad_class": None,
        "model_class_kwargs": {},
        "filter_caps": ["completion"],
    },
    "wizardlm/7b": {
        "backend": "autogptq",
        "model_path": "TheBloke/WizardLM-7B-V1.0-Uncensored-GPTQ",
        "diff_scratchpad_class": None,
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceWizard",
        "model_class_kwargs": {
            "model_basename": "wizardlm-7b-v1.0-uncensored-GPTQ-4bit-128g.no-act.order",
        },
        "filter_caps": ["wizardlm"],
    },
    "wizardlm/13b": {
        "backend": "autogptq",
        "model_path": "TheBloke/WizardLM-13B-V1.1-GPTQ",
        "diff_scratchpad_class": None,
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceWizard",
        "model_class_kwargs": {
            "model_basename": "wizardlm-13b-v1.1-GPTQ-4bit-128g.no-act.order",
        },
        "filter_caps": ["wizardlm"],
    },
    "llama2/7b": {
        "backend": "autogptq",
        "model_path": "TheBloke/Llama-2-7b-Chat-GPTQ",
        "diff_scratchpad_class": None,
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceLlama2",
        "model_class_kwargs": {
            "model_basename": "gptq_model-4bit-128g",
        },
        "filter_caps": ["llama2"],
    },
    "llama2/13b": {
        "backend": "autogptq",
        "model_path": "TheBloke/Llama-2-13B-chat-GPTQ",
        "diff_scratchpad_class": None,
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceLlama2",
        "model_class_kwargs": {
            "model_basename": "gptq_model-4bit-128g",
        },
        "filter_caps": ["llama2"],
    },
    "wizardlm/30b/4bit": {
        "backend": "transformers",
        "model_path": "TheBloke/WizardLM-30B-fp16",
        "diff_scratchpad_class": None,
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceWizard",
        "model_class_kwargs": {
            "load_in_4bit": True,
        },
        "filter_caps": ["wizardlm"],
        "hidden": True,   # only for debugging because wip on sharding
     },
}
