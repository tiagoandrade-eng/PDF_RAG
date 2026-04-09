from __future__ import annotations

import threading


class TokenCounter:
    PRECOS = {
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
        "gpt-4o": {"input": 0.0025, "output": 0.010},
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tokens: dict[str, dict[str, int]] = {}

    def add(self, modelo: str, input_tokens: int, output_tokens: int = 0) -> None:
        with self._lock:
            entry = self.tokens.setdefault(modelo, {"input": 0, "output": 0})
            entry["input"] += input_tokens
            entry["output"] += output_tokens

    def custo_usd(self) -> float:
        total = 0.0
        for modelo, contagem in self.tokens.items():
            preco = self.PRECOS.get(modelo, {"input": 0.0, "output": 0.0})
            total += (contagem["input"] / 1000) * preco["input"]
            total += (contagem["output"] / 1000) * preco.get("output", 0.0)
        return total

    def resumo(self) -> str:
        if not self.tokens:
            return "   Tokens: nenhuma chamada registrada"

        linhas = ["   -- Tokens OpenAI ---------------------"]
        total_input = 0
        total_output = 0

        for modelo, contagem in sorted(self.tokens.items()):
            input_tokens = contagem["input"]
            output_tokens = contagem["output"]
            total_input += input_tokens
            total_output += output_tokens
            preco = self.PRECOS.get(modelo, {"input": 0.0, "output": 0.0})
            custo = (input_tokens / 1000) * preco["input"]
            custo += (output_tokens / 1000) * preco.get("output", 0.0)
            linhas.append(
                f"   {modelo:<28} in={input_tokens:>8,}  out={output_tokens:>7,}  ~${custo:.4f}"
            )

        linhas.append(
            f"   {'TOTAL':<28} in={total_input:>8,}  out={total_output:>7,}  ~${self.custo_usd():.4f}"
        )
        linhas.append("   ----------------------------------------")
        return "\n".join(linhas)
