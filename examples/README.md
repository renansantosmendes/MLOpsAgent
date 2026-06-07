# MLOpsAgent - Exemplos e Conceitos

Este diretório contém exemplos educacionais e comparações arquiteturais.

## 📚 Arquivos

### [`comparison_nodes_vs_tools.py`](comparison_nodes_vs_tools.py)
**Comparação executável entre Nodes e Tools**

Demonstra lado a lado:
- ✅ **StateGraph com Nodes** (abordagem atual)
- 🤖 **LLM Agent com Tools** (abordagem alternativa)

**Execute:**
```bash
python examples/comparison_nodes_vs_tools.py
```

**Você verá:**
- Fluxo de execução de cada abordagem
- Diferenças de performance
- Quando usar cada uma

---

### [`agent_with_tools_example.py`](agent_with_tools_example.py)
**Exemplo conceitual de agent com @tool decorators**

Mostra como seria a implementação se usássemos:
```python
@tool
def process_incoming_data(...):
    """Tool que o LLM pode chamar"""
    ...

@tool
def detect_data_drift(...):
    """Outra tool disponível para o LLM"""
    ...
```

**Leia o arquivo para entender:**
- Como funciona o decorator `@tool`
- Quando usar LLM reasoning
- Diferença entre tools e nodes

---

## 🎯 Pergunta Frequente

### "Por que não usamos `@tool` no MLOpsAgent?"

**Resposta rápida:** Porque nosso agente usa **Nodes** em um **StateGraph**, não **Tools** para um **LLM Agent**.

| Aspecto | Nodes (Atual) | Tools (Alternativa) |
|---------|---------------|---------------------|
| **Controle** | Regras determinísticas | LLM decide |
| **Performance** | ⚡ Rápido (sem LLM) | 🐌 Lento |
| **Custo** | Grátis | $$$ (tokens) |
| **Previsibilidade** | 100% | ~95% |

**Para detalhes completos, veja:** [`../ARCHITECTURE.md`](../ARCHITECTURE.md)

---

## 🔍 Quando Usar Cada Abordagem

### Use **Nodes** (StateGraph) quando:
- ✅ Pipeline tem fluxo bem definido
- ✅ Decisões baseadas em métricas/thresholds
- ✅ Performance e custo são importantes
- ✅ Comportamento deve ser previsível

**Exemplo:** Pipeline automatizado de retreino de ML ← **Nosso caso!**

### Use **Tools** (LLM Agent) quando:
- 🤖 Precisa de reasoning sobre estratégias
- 🤖 Fluxo varia muito por contexto  
- 🤖 Tarefas exploratórias são comuns
- 🤖 Custo de LLM é justificável

**Exemplo:** "Analise por que o modelo está ruim e sugira ações"

---

## 📖 Recursos Adicionais

### Documentação do Projeto
- [`../README.md`](../README.md) - Overview do projeto
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) - Decisões arquiteturais
- [`../REFACTORING_SUMMARY.md`](../REFACTORING_SUMMARY.md) - Resumo da refatoração

### Código Principal
- [`../mlops_agent/graph_agent.py`](../mlops_agent/graph_agent.py) - Implementação do StateGraph
- [`../mlops_agent/ingestion.py`](../mlops_agent/ingestion.py) - Tools de ingestão
- [`../mlops_agent/training.py`](../mlops_agent/training.py) - Tools de treinamento

### LangGraph Docs
- [StateGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [Tools vs Nodes](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
- [ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)

---

## 💡 Dica

**Confuso sobre Nodes vs Tools?**

Execute o exemplo de comparação:
```bash
python examples/comparison_nodes_vs_tools.py
```

Você verá exatamente como cada abordagem funciona! 🚀
