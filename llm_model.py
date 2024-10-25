from ollama import LLM
from ollama.some_module import LLM


def load_llm_model(model_path='phi-2.Q4_K_M.gguf'):
    llm = LLM(
        model_path=model_path,
        n_gpu_layers=40,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        verbose=True,
    )
    return llm
