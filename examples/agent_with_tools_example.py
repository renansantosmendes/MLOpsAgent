"""
Exemplo CONCEITUAL de como seria se usássemos @tool com LLM reasoning.
Este é apenas um exemplo educacional - não está integrado ao código principal.
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import pandas as pd


# ============================================================================
# CENÁRIO 1: Nodes do StateGraph (ATUAL - sem @tool)
# ============================================================================
# Usado quando você tem um FLUXO FIXO e DECISÕES BASEADAS EM REGRAS

def node_data_process(state: dict) -> dict:
    """Node do grafo - executa sempre na sequência definida."""
    result = data_process_tool(state["input_source"], state["reference_dataset_path"])
    return {**state, **result}

# O grafo decide a próxima etapa baseado em condições:
# if state["has_new_records"]: goto "drift_check"
# else: goto "halt"


# ============================================================================
# CENÁRIO 2: Tools com LLM reasoning (ALTERNATIVA - com @tool)
# ============================================================================
# Usado quando você quer que um LLM DECIDA quais ferramentas chamar

@tool
def process_incoming_data(input_source: str, reference_path: str) -> dict:
    """
    Process incoming data and compare against reference dataset.
    
    Args:
        input_source: URL or path to new data
        reference_path: Path to reference dataset
        
    Returns:
        Dictionary with processed data and flags
    """
    from mlops_agent.ingestion import data_process_tool
    return data_process_tool(input_source, reference_path)


@tool
def detect_data_drift(reference_data_json: str, new_data_json: str) -> dict:
    """
    Detect data drift between reference and new datasets.
    
    Args:
        reference_data_json: JSON string of reference data
        new_data_json: JSON string of new data
        
    Returns:
        Dictionary with drift detection results
    """
    import pandas as pd
    from mlops_agent.drift import data_drift_tool
    
    ref_df = pd.read_json(reference_data_json)
    new_df = pd.read_json(new_data_json)
    return data_drift_tool(ref_df, new_df)


@tool
def train_models(data_json: str, target_column: str) -> dict:
    """
    Train multiple candidate models on the provided dataset.
    
    Args:
        data_json: JSON string of training data
        target_column: Name of target column
        
    Returns:
        Dictionary with training results and best model info
    """
    import pandas as pd
    from mlops_agent.training import model_training_tool, model_selection_tool
    
    data = pd.read_json(data_json)
    training_result = model_training_tool(data, target_column)
    selection_result = model_selection_tool(training_result["candidate_results"])
    return selection_result


# Criar agente com LLM que pode escolher usar essas tools
def create_mlops_reasoning_agent():
    """
    Cria um agente que usa LLM para decidir quais ferramentas usar.
    O LLM recebe a tarefa e decide autonomamente:
    - Quando processar dados
    - Quando checar drift
    - Quando treinar modelos
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    tools = [
        process_incoming_data,
        detect_data_drift, 
        train_models,
    ]
    
    # ReAct agent: LLM que usa reasoning para escolher tools
    agent = create_react_agent(llm, tools)
    
    return agent


# ============================================================================
# COMPARAÇÃO
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ NODES (Atual)                  │ TOOLS (Alternativa)                    │
├────────────────────────────────┼────────────────────────────────────────┤
│ ✅ Fluxo determinístico        │ 🤖 LLM decide o fluxo                  │
│ ✅ Rápido (sem LLM calls)      │ ⚠️  Lento (muitas chamadas LLM)        │
│ ✅ Previsível                  │ ⚠️  Imprevisível                        │
│ ✅ Barato                      │ 💰 Caro (tokens LLM)                   │
│ ⚠️  Rígido                     │ ✅ Flexível                            │
│ ⚠️  Sem reasoning              │ ✅ Pode adaptar estratégia             │
└────────────────────────────────┴────────────────────────────────────────┘

QUANDO USAR NODES (atual):
- Pipeline de ML bem definido
- Decisões baseadas em métricas/thresholds
- Performance e custo são importantes
- Fluxo previsível é desejado

QUANDO USAR TOOLS (@tool):
- Tarefas complexas que requerem reasoning
- Fluxo pode mudar baseado no contexto
- Precisa de adaptabilidade dinâmica
- Exemplo: "Analise esses dados e decida se devemos retreinar"
"""


# ============================================================================
# EXEMPLO DE USO COM TOOLS
# ============================================================================

def example_with_llm_tools():
    """
    Exemplo de como o LLM usaria as tools:
    
    User: "Processar novos dados de vendas e retreinar se houver drift"
    
    LLM Reasoning:
    1. Thought: Preciso primeiro processar os dados novos
       Action: process_incoming_data(...)
    
    2. Thought: Agora verificar se há drift
       Action: detect_data_drift(...)
    
    3. Thought: Drift detectado! Preciso retreinar
       Action: train_models(...)
    
    4. Final Answer: Modelo retreinado com F1=0.92
    """
    agent = create_mlops_reasoning_agent()
    
    result = agent.invoke({
        "messages": [
            ("user", "Processar dados de https://example.com/new_sales.csv e retreinar se necessário")
        ]
    })
    
    return result


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("RESUMO:")
    print("="*80)
    print("""
    Nosso MLOpsAgent usa NODES porque:
    
    ✅ Não precisamos de um LLM decidindo o fluxo
    ✅ O fluxo é bem definido: ingest → drift → train → deploy
    ✅ Decisões são baseadas em regras (if drift > threshold: retrain)
    ✅ Mais rápido, barato e previsível
    
    Usaríamos @tool se:
    
    🤖 Quiséssemos que um LLM decidisse quando retreinar
    🤖 O fluxo precisasse adaptar baseado em contexto
    🤖 Houvesse tarefas complexas requerendo reasoning
    """)
