# FAQ - Perguntas Frequentes

## 🤔 Por que não foi usado o decorator `@tool` nas tools do agente?

### Resposta Curta

Porque **estas não são "tools" no sentido de LangChain/LangGraph**, mas sim **nodes** (nós) de um StateGraph.

- **Tools** (`@tool`) → Ferramentas que um **LLM escolhe** chamar
- **Nodes** (sem `@tool`) → Funções de **processamento no fluxo fixo** do grafo

### Resposta Detalhada

#### O que temos no código atual:

```python
# NODES - Funções de processamento (SEM @tool)
def node_data_process(state: dict) -> dict:
    result = data_process_tool(...)
    return {**state, **result}

def node_data_drift(state: dict) -> dict:
    result = data_drift_tool(...)
    return {**state, **result}

# Fluxo determinístico
START → data_process → drift → train → deploy → END
```

#### O que seria se usássemos `@tool`:

```python
from langchain.tools import tool

# TOOLS - Ferramentas para o LLM (COM @tool)
@tool
def process_incoming_data(input_source: str) -> dict:
    """Process new data - LLM can choose to call this"""
    ...

@tool
def detect_data_drift(data: str) -> dict:
    """Detect drift - LLM can choose to call this"""
    ...

# LLM decide quando chamar cada tool
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools=[process_incoming_data, detect_data_drift])
```

### Comparação Lado a Lado

| Aspecto | Nodes (Atual) | Tools (Alternativa) |
|---------|---------------|---------------------|
| **Quem decide o fluxo?** | Regras de código | LLM reasoning |
| **Chamadas LLM** | 0 | 3-10 por execução |
| **Latência** | ~segundos | ~minutos |
| **Custo** | $0 | $0.01-0.10 por run |
| **Previsibilidade** | 100% | ~95% |
| **Debugging** | Fácil | Difícil |
| **Quando usar?** | Pipeline bem definido | Tarefas exploratórias |

### Por que escolhemos Nodes?

✅ **Nosso caso de uso é perfeito para Nodes:**

1. **Pipeline bem definido**: ingest → drift → train → deploy
2. **Decisões baseadas em regras**: `if drift > 0.2: retrain`
3. **Performance crítica**: retreino deve ser rápido
4. **Custo previsível**: sem surpresas de billing
5. **Produção robusta**: comportamento consistente

⚠️  **Tools seriam úteis SE precisássemos de:**

1. LLM decidindo qual estratégia de retreino usar
2. Análise exploratória de por que modelo falhou
3. Adaptação dinâmica baseado em contexto complexo
4. Reasoning sobre trade-offs (custo vs performance)

### Exemplo Prático

**Com Nodes (atual):**
```
Tempo: 10 segundos
Custo: $0
Caminho: data_process → drift (detectado) → train → deploy
```

**Com Tools (alternativa):**
```
Tempo: 60 segundos
Custo: $0.05
Caminho: LLM pensa → chama process_data → LLM pensa → 
         chama check_drift → LLM pensa → chama train_model → 
         LLM pensa → responde
```

### Recursos para Aprender Mais

- 📖 [ARCHITECTURE.md](ARCHITECTURE.md) - Decisão arquitetural completa
- 💻 [examples/comparison_nodes_vs_tools.py](examples/comparison_nodes_vs_tools.py) - Código executável
- 📚 [examples/README.md](examples/README.md) - Guia dos exemplos

---

## 🔄 Quando deveria usar cada abordagem?

### Use **Nodes** (StateGraph) quando:

```python
# Pipeline determinístico
if drift_detected and f1_score < threshold:
    retrain_model()  # Decisão clara e objetiva
```

**Casos de uso:**
- ✅ Pipeline de ML automatizado
- ✅ ETL workflows
- ✅ Processamento de dados em batch
- ✅ Qualquer fluxo com regras bem definidas

### Use **Tools** (LLM Agent) quando:

```python
# Decisão que requer reasoning
"O modelo está com F1=0.65, drift=0.15, custo de retreino=$500.
 Devo retreinar agora ou esperar mais dados?"
# → LLM analisa contexto e decide
```

**Casos de uso:**
- 🤖 Análise exploratória de dados
- 🤖 Debugging de modelos complexos
- 🤖 Decisões que requerem trade-offs
- 🤖 Prototipagem rápida de soluções

---

## 📊 Exemplo de Execução

Execute para ver a diferença:

```bash
python examples/comparison_nodes_vs_tools.py
```

Output mostrará:
```
ABORDAGEM 1: StateGraph com Nodes
  [NODE] Verificando drift...
  → Drift detectado: True
  [NODE] Treinando modelo...
  → Modelo treinado com F1=0.92
  ✅ Resultado: model_trained
  Tempo: 0.1s | Custo: $0

ABORDAGEM 2: LLM Agent com Tools
  💭 LLM Thought: I need to check drift first
  [TOOL] check_data_drift(drift_share=0.25)
  💭 LLM Thought: Drift detected! Should train
  [TOOL] train_ml_model(reason='drift detected')
  ✅ LLM Final Answer: Model trained
  Tempo: 15s | Custo: $0.03
```

---

## 🎯 Resumo Executivo

**Não usamos `@tool` porque:**

1. Nosso agente tem **fluxo determinístico** (StateGraph)
2. Não precisamos de **LLM reasoning**
3. Queremos **performance** e **custo baixo**
4. Precisamos de **comportamento previsível**

**Esta é a escolha CORRETA para nosso caso de uso!** ✅

Se no futuro precisarmos de reasoning complexo, podemos adicionar tools para casos específicos enquanto mantemos o core do pipeline com nodes.

---

*Dúvidas? Veja os exemplos em [`examples/`](examples/) ou leia [ARCHITECTURE.md](ARCHITECTURE.md)*
