# CHANGELOG - Análise Neurofisiológica com Suporte a LSL

## v1.3 - Integração com LSL (2026-04-25)

### ✨ Novas Funcionalidades

1. **Suporte a Streams LSL**
   - Captura de dois streams: "EMG" (sinal bruto) e "EMG_Processado" (filtrado)
   - Integração automática com detectores de streams LSL
   - Timeout configurável (3 segundos por padrão)

2. **Botão de Recarregar**
   - Novo botão "⬇️ RECARREGAR DE LSL" na sidebar
   - Captura 5 segundos de dados de streams ativos
   - Regenera gráficos automaticamente

3. **Simulador LSL**
   - Novo arquivo `simulador_lsl_emg.py`
   - Gera dois streams LSL com dados sintéticos realistas
   - Suporta transmissão contínua ou limitada por tempo
   - Configurável: frequência, canais, intervalo

4. **Processamento de Dados Dinâmicos**
   - Método `_criar_raw_mne()` para converter dados LSL em estrutura MNE
   - Suporte a múltiplos canais com nomes automáticos
   - Filtragem automática (passa-banda 1-40 Hz)

### 🐛 Correções

1. **Matplotlib 3.9+ Compatibility**
   - Alterado `labels=` para `tick_labels=` em `ax.boxplot()`
   - Removidos parâmetros `pad` inválidos de `set_xticklabels()`
   - Movido `pad` para `tick_params()` (local correto)

2. **Parâmetros R (par)**
   - Corrigido `par(usr)` para `par(usr = old_usr)` em função de painel
   - Restauração correta de parâmetros gráficos R

3. **Espaçamento de Labels**
   - Aumentada rotação de labels X: 12° → 35°
   - Adicionado padding em labels e ticks
   - Melhorada legibilidade em gráficos radar e matriz de dispersão

### 📦 Novos Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `simulador_lsl_emg.py` | Simulador de streams LSL com dados sintéticos |
| `teste_lsl_integracao.py` | Script de teste de integração |
| `testes_unitarios.py` | Suite de testes unitários |
| `README_ANALISE_LSL.md` | Documentação completa |
| `QUICKSTART.txt` | Guia rápido de instalação |
| `requirements.txt` | Dependências Python |
| `CHANGELOG.md` | Este arquivo |

### 🔧 Melhorias Técnicas

1. **Gestão de Erros**
   - Try-catch abrangente para operações LSL
   - Feedback visual ao usuário via QMessageBox
   - Logging detalhado de cada etapa

2. **Performance**
   - Captura de dados otimizada (timeout de 0.1s)
   - Processamento em thread principal com feedback visual
   - Caching de streams LSL

3. **Estrutura de Código**
   - Separação clara: simulador, aplicação, testes
   - Métodos bem documentados
   - Código modular e reutilizável

### 📊 Métricas Calculadas

Mantidas as 4 métricas principais:
- **RMS**: Root Mean Square
- **Freq. Mediana**: Frequência central da distribuição de potência
- **ZCR**: Zero Crossing Rate
- **Waveform Length**: Comprimento cumulativo

Com classificação em modos:
- Baixa energia (Q1)
- Média energia (Q2)
- Alta energia (Q3+)

### 🎨 Interface

Mantida interface PyQt6 com novos elementos:
- Novo botão "⬇️ RECARREGAR DE LSL" na sidebar
- Feedback visual via status bar
- Mensagens de erro e aviso intuitivas

### 📋 Dependências Adicionadas

```
pylsl==1.16.0          # Lab Streaming Layer
```

Todas as outras mantidas compatíveis com versões anteriores.

### 🧪 Testes

- Suite de testes unitários criada
- Testes de captura, filtragem e classificação
- Script de integração para validação de todo o fluxo
- Execução: `python testes_unitarios.py`

### 📝 Documentação

- README completo com arquitetura
- Guia de troubleshooting
- Exemplos de uso
- Diagrama de fluxo de dados

### 🚀 Como Usar

```bash
# Terminal 1: Simulador
python simulador_lsl_emg.py

# Terminal 2: Aplicação
python Grafico.py

# Na interface: Clique "⬇️ RECARREGAR DE LSL"
```

### ⚠️ Notas Importantes

- LSL é opcional - aplicação funciona sem streams
- R é opcional - matplotlib é fallback automático
- Gráficos em alta resolução: 300 DPI

### 🔄 Compatibilidade

- ✅ Python 3.8+
- ✅ Matplotlib 3.9+
- ✅ PyQt6 6.7+
- ✅ MNE 1.6+
- ✅ Windows, Linux, macOS

### 🎯 Próximas Melhorias Sugeridas

1. Modo de gravação contínua de streams
2. Visualização em tempo real dos dados capturados
3. Configuração de parâmetros de filtragem na UI
4. Exportação em múltiplos formatos (CSV, Excel, PDF)
5. Cache de dados para comparação histórica

---

**Versão**: 1.3 Beta  
**Data**: 2026-04-25  
**Status**: Pronto para teste
