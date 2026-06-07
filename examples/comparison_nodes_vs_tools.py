"""
Comparação Prática: Nodes (StateGraph) vs Tools (LLM Agent)

Este arquivo demonstra a diferença entre as duas abordagens
com exemplos executáveis.

NOTA: Esta versão é simplificada para rodar sem dependências pesadas.
O código real em mlops_agent/graph_agent.py usa LangGraph completo.
"""

import pandas as pd
from typing import Any, Literal


# =============================================================================
# ABORDAGEM 1: NODES com StateGraph (ATUAL - Determinístico)
# =============================================================================

print("="*80)
print("ABORDAGEM 1: StateGraph com Nodes (Determinístico)")
print("="*80)

def approach_1_stategraph():
    """
    Nodes = Funções de processamento em um grafo de estados
    - Fluxo fixo e previsível
    - Decisões baseadas em regras
    - Sem LLM envolvido
    
    NOTA: Esta é uma versão simplificada sem LangGraph para demonstração.
    O código real em mlops_agent/graph_agent.py usa LangGraph completo.
    """
    
    # Definir estado (versão simplificada)
    state: dict[str, Any] = {
        "drift_share": 0.25,
        "has_new_records": True,
    }
    
    # NODES - Funções simples de processamento
    def check_drift_node(state: dict) -> dict:
        """Node que verifica drift - sem @tool decorator"""
        print("  [NODE] Verificando drift...")
        # Lógica determinística
        state["drift_detected"] = state.get("drift_share", 0.0) > 0.2
        print(f"  → Drift detectado: {state['drift_detected']}")
        return state
    
    def train_model_node(state: dict) -> dict:
        """Node que treina modelo - sem @tool decorator"""
        print("  [NODE] Treinando modelo...")
        state["f1_score"] = 0.92
        state["action_taken"] = "model_trained"
        print(f"  → Modelo treinado com F1={state['f1_score']}")
        return state
    
    def halt_node(state: dict) -> dict:
        """Node que para o processo - sem @tool decorator"""
        print("  [NODE] Nenhuma ação necessária")
        state["action_taken"] = "halted"
        return state
    
    # Roteamento condicional (regras de negócio)
    def route_after_drift(state: dict) -> Literal["train", "halt"]:
        """Decisão baseada em REGRAS, não em LLM"""
        if state.get("drift_detected") or state.get("has_new_records"):
            return "train"
        return "halt"
    
    # Executar fluxo manualmente (simulando StateGraph)
    print("\n📊 Executando StateGraph (simulado)...")
    
    # 1. Check drift
    state = check_drift_node(state)
    
    # 2. Route baseado em regras
    next_node = route_after_drift(state)
    print(f"  [ROUTE] Decisão: ir para '{next_node}'")
    
    # 3. Executar próximo node
    if next_node == "train":
        state = train_model_node(state)
    else:
        state = halt_node(state)
    
    print(f"\n✅ Resultado: {state['action_taken']}")
    
    return state


# =============================================================================
# ABORDAGEM 2: TOOLS com LLM Agent (Alternativa - Reasoning)
# =============================================================================

print("\n" + "="*80)
print("ABORDAGEM 2: LLM Agent com Tools (Reasoning)")
print("="*80)

def approach_2_llm_tools():
    """
    Tools = Ferramentas que o LLM pode escolher usar
    - Fluxo dinâmico (LLM decide)
    - Decisões baseadas em reasoning
    - LLM escolhe qual tool chamar
    
    NOTA: Este é um exemplo CONCEITUAL (mock sem LangChain)
    Em produção real, você usaria @tool do langchain.tools
    """
    
    # TOOLS - Simulando o que seria com @tool decorator
    # Em produção: from langchain.tools import tool + @tool decorator
    def check_data_drift(drift_share: float, threshold: float = 0.2) -> dict:
        """
        Check if data drift exceeds threshold.
        
        Args:
            drift_share: Share of drifted features (0.0 to 1.0)
            threshold: Drift threshold (default 0.2)
            
        Returns:
            Dictionary with drift status
        """
        print(f"  [TOOL] check_data_drift(drift_share={drift_share}, threshold={threshold})")
        drift_detected = drift_share > threshold
        return {
            "drift_detected": drift_detected,
            "message": f"Drift {'DETECTED' if drift_detected else 'not detected'}"
        }
    
    def train_ml_model(reason: str) -> dict:
        """
        Train a machine learning model.
        
        Args:
            reason: Reason for training (e.g., "drift detected", "new data")
            
        Returns:
            Dictionary with training results
        """
        print(f"  [TOOL] train_ml_model(reason='{reason}')")
        return {
            "f1_score": 0.92,
            "status": "success",
            "message": f"Model trained successfully. Reason: {reason}"
        }
    
    def skip_retraining(reason: str) -> dict:
        """
        Skip model retraining.
        
        Args:
            reason: Reason for skipping
            
        Returns:
            Dictionary with skip status
        """
        print(f"  [TOOL] skip_retraining(reason='{reason}')")
        return {
            "status": "skipped",
            "message": f"Retraining skipped. Reason: {reason}"
        }
    
    # Em produção real, você usaria:
    # from langchain_openai import ChatOpenAI
    # from langgraph.prebuilt import create_react_agent
    # 
    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    # agent = create_react_agent(llm, tools=[check_data_drift, train_ml_model, skip_retraining])
    # 
    # result = agent.invoke({
    #     "messages": [("user", "Drift share is 0.25, should I retrain?")]
    # })
    
    # MOCK: Simulando o que o LLM faria
    print("\n🤖 Simulando LLM Agent reasoning...")
    print("\n💭 LLM Thought: I need to check if there's data drift first")
    drift_result = check_data_drift(drift_share=0.25)
    
    print("\n💭 LLM Thought: Drift detected! I should train the model")
    train_result = train_ml_model(reason="Data drift detected")
    
    print(f"\n✅ LLM Final Answer: {train_result['message']}")
    
    return train_result


# =============================================================================
# COMPARAÇÃO DE EXECUÇÃO
# =============================================================================

print("\n" + "="*80)
print("EXECUTANDO AMBAS ABORDAGENS")
print("="*80)

print("\n🔹 Cenário: drift_share=0.25, has_new_records=True")
print("-" * 80)

# Executar Abordagem 1
result1 = approach_1_stategraph()

# Executar Abordagem 2
result2 = approach_2_llm_tools()

# Comparação
print("\n" + "="*80)
print("COMPARAÇÃO DE RESULTADOS")
print("="*80)

comparison = """
┌─────────────────────────────────────────────────────────────────────────┐
│                    StateGraph (Nodes)  │  LLM Agent (Tools)             │
├────────────────────────────────────────┼────────────────────────────────┤
│ Fluxo de Execução:                     │                                │
│   1. check_drift_node()                │   1. LLM pensa                 │
│   2. route_after_drift()               │   2. LLM chama check_data_drift│
│   3. train_model_node()                │   3. LLM pensa novamente       │
│                                        │   4. LLM chama train_ml_model  │
│                                        │   5. LLM formula resposta      │
├────────────────────────────────────────┼────────────────────────────────┤
│ Chamadas LLM:           0              │   3-5 (reasoning + tool calls) │
│ Latência:               ~0.1s          │   ~5-15s                       │
│ Custo:                  $0             │   ~$0.01-0.05                  │
│ Previsibilidade:        100%           │   ~95%                         │
│ Debugging:              Fácil          │   Médio/Difícil                │
├────────────────────────────────────────┼────────────────────────────────┤
│ QUANDO USAR:                           │                                │
│ ✅ Pipeline ML bem definido            │ ✅ Decisões complexas          │
│ ✅ Regras claras                       │ ✅ Contexto variável           │
│ ✅ Performance crítica                 │ ✅ Tarefas exploratórias       │
│ ✅ Produção automatizada               │ ✅ Prototipagem rápida         │
└────────────────────────────────────────┴────────────────────────────────┘
"""

print(comparison)

# =============================================================================
# CONCLUSÃO
# =============================================================================

print("\n" + "="*80)
print("CONCLUSÃO")
print("="*80)

conclusion = """
Para MLOpsAgent, escolhemos StateGraph com Nodes porque:

✅ VANTAGENS NO NOSSO CASO:
   • Pipeline de ML tem fluxo bem definido
   • Decisões baseadas em métricas objetivas (F1, drift_share, etc)
   • Performance é crítica para produção
   • Custo deve ser previsível
   • Debugging precisa ser simples
   • Compliance e auditoria requerem determinismo

⚠️  TOOLS seriam úteis SE:
   • Precisássemos de reasoning sobre qual estratégia usar
   • Fluxo variasse muito por contexto
   • Tarefas fossem exploratórias (análise de dados)
   • Custo de LLM fosse justificável
   • Latência adicional fosse aceitável

🎯 RECOMENDAÇÃO:
   Mantenha Nodes para o fluxo principal de retreino.
   Considere adicionar Tools apenas para:
   - Análise exploratória de problemas
   - Decisões que requerem contexto complexo
   - Features experimentais
"""

print(conclusion)

if __name__ == "__main__":
    print("\n✨ Exemplo executado com sucesso!")
    print("\nPara ver o código real do StateGraph:")
    print("  → mlops_agent/graph_agent.py")
    print("\nPara exemplo conceitual com Tools:")
    print("  → examples/agent_with_tools_example.py")
