import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ==========================================
# 6. GERADOR DE GRÁFICOS
# ==========================================
def gerar_graficos(df):
    print("\n📊 A gerar o gráfico comparativo...")
    categorias = ['correta', 'parcialmente_correta', 'fora_de_escopo', 'insegura', 'alucinacao']
    
    base_counts = df['baseline_classificacao'].value_counts().reindex(categorias, fill_value=0)
    tide_counts = df['tide_classificacao'].value_counts().reindex(categorias, fill_value=0)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categorias))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, base_counts, width, label='Baseline (Sem RAG)', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x + width/2, tide_counts, width, label='Tide (Com RAG)', color='#66b3ff', edgecolor='black')
    
    ax.set_ylabel('Número de Respostas', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Qualidade: Modelo Baseline vs RAG (Tide)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Correta', 'Parcial. Correta', 'Fora de Escopo', 'Insegura', 'Alucinação'], fontsize=11)
    ax.legend(fontsize=11)
    
    ax.bar_label(rects1, padding=3, fontsize=11)
    ax.bar_label(rects2, padding=3, fontsize=11)
    
    fig.tight_layout()
    nome_grafico = "comparacao_modelos.png"
    plt.savefig(nome_grafico, dpi=300)
    print(f"✅ Gráfico guardado com sucesso como '{nome_grafico}'!")

# ==========================================
# 7. GERADOR DE PARÁGRAFO ACADÉMICO
# ==========================================
def imprimir_paragrafo_conclusao(df, total_perguntas):
    base_verif = len(df[df['baseline_classificacao'] == 'correta'])
    base_parcial = len(df[df['baseline_classificacao'] == 'parcialmente_correta'])
    base_inseg = len(df[df['baseline_classificacao'] == 'insegura'])
    base_aluc = len(df[df['baseline_classificacao'] == 'alucinacao'])
    
    tide_verif = len(df[df['tide_classificacao'] == 'correta'])
    tide_parcial = len(df[df['tide_classificacao'] == 'parcialmente_correta'])
    tide_inseg = len(df[df['tide_classificacao'] == 'insegura'])
    tide_aluc = len(df[df['tide_classificacao'] == 'alucinacao'])
    
    acc_base = (base_verif / total_perguntas) * 100 if total_perguntas > 0 else 0
    acc_tide = (tide_verif / total_perguntas) * 100 if total_perguntas > 0 else 0
    
    texto_artigo = (
        f"Para validar a segurança e precisão das respostas, adotou-se o paradigma LLM-as-a-judge "
        f"com separação estrita entre falhas de aterramento (alucinações) e falhas de alinhamento médico (respostas inseguras). "
        f"O modelo de uso geral (baseline) obteve {acc_base:.1f}% de respostas totalmente corretas ({base_verif} em {total_perguntas}), "
        f"registando {base_parcial} resposta(s) parcialmente correta(s), {base_inseg} caso(s) de infração de segurança médica (prescrições ou erros conceituais) "
        f"e {base_aluc} caso(s) classificado(s) como alucinação. "
        f"Em contrapartida, o sistema proposto (Tide) com arquitetura RAG alcançou {acc_tide:.1f}% de respostas corretas "
        f"({tide_verif} ocorrências), com {tide_parcial} resposta(s) parcialmente correta(s), {tide_inseg} resposta(s) insegura(s) "
        f"e {tide_aluc} incidência(s) de alucinação, permitindo isolar a eficácia da recuperação de contexto frente ao alinhamento de segurança."
    )
    
    print("\n" + "="*70)
    print("📋 TEXTO GERADO PARA O SEU ARTIGO/DISSERTAÇÃO:")
    print("="*70)
    print(texto_artigo.replace('.', ',')) 
    print("="*70 + "\n")


nome_ficheiro = "resultados_avaliacao_Teste final.csv"
df_resultados = pd.read_csv(nome_ficheiro, encoding="utf-8")

gerar_graficos(df_resultados)
imprimir_paragrafo_conclusao(df_resultados, 100)