# Análise Neurofisiológica com Suporte a LSL

## Overview
Sistema de análise em tempo real de sinais EMG com captura de dados via LSL (Lab Streaming Layer) e geração de gráficos estatísticos usando R (ggplot2) ou fallback em Python (matplotlib).

## Requisitos

### Dependências Python
```bash
pip install PyQt6 PyQt6-WebEngine pyqtgraph mne numpy scipy pandas seaborn matplotlib pylsl
```

### Dependências R (opcional, para melhor qualidade gráfica)
```R
install.packages(c("ggplot2", "tidyr", "dplyr", "GGally", "scales"))
```

## Uso

### 1. Iniciar o Simulador LSL (Terminal 1)

Transmissão por 30 segundos:
```bash
python simulador_lsl_emg.py --duracao 30 --sfreq 250 --canais 4 --intervalo 0.1
```

Transmissão contínua:
```bash
python simulador_lsl_emg.py --sfreq 250 --canais 4 --intervalo 0.1
```

**Opções:**
- `--duracao`: Duração em segundos (padrão: contínuo)
- `--sfreq`: Frequência de amostragem em Hz (padrão: 250)
- `--canais`: Número de canais EMG (padrão: 4)
- `--intervalo`: Intervalo entre buffers em segundos (padrão: 0.1)

### 2. Executar a Aplicação Principal (Terminal 2)

```bash
python Grafico.py
```

### 3. Usar o Botão "⬇️ RECARREGAR DE LSL"

Clique no botão na sidebar para:
1. Procurar streams LSL disponíveis ("EMG" e "EMG_Processado")
2. Capturar 5 segundos de dados
3. Processar e gerar gráficos automaticamente

## Funcionalidades

### Gráficos Disponíveis

1. **Sinal Bruto (EEG)** - Visualização pyqtgraph do sinal original
2. **Sinal Filtrado (Passa-Banda 1-40Hz)** - Sinal após filtragem
3. **Análise Espectral e Estatística** - 3 gráficos (R + ggplot2 ou matplotlib):
   - **Boxplot Facetado**: Distribuição de features por modo de energia
   - **Matriz de Dispersão**: Correlação entre features
   - **Radar Chart**: Perfil normalizado das features

### Modos de Operação

- **Modo Tabela**: Visualiza métricas em tabela (RMS, Freq. Mediana, ZCR, Waveform Length)
- **Modo Gráfico**: Visualiza análise estatística gerada por R/Python

### Botões na Sidebar

- **📊 DASHBOARD PRINCIPAL**: Em desenvolvimento
- **🔍 INSPEÇÃO DE CANAIS**: Em desenvolvimento
- **💾 EXPORTAR RELATÓRIO**: Em desenvolvimento
- **🧮 ALTERNAR VISUALIZAÇÃO**: Alterna entre tabela e gráficos
- **🔄 RESETAR ZOOM**: Reset do zoom nos gráficos de sinal
- **⬇️ RECARREGAR DE LSL**: Captura dados de LSL e regenera gráficos

## Streams LSL

### Stream "EMG" (Bruto)
- **Nome**: EMG
- **Tipo**: EMG
- **Formato**: float32
- **Frequência**: 250 Hz (configurável)
- **Canais**: 4 (configurável)

### Stream "EMG_Processado" (Filtrado)
- **Nome**: EMG_Processado
- **Tipo**: EMG
- **Formato**: float32
- **Frequência**: 250 Hz (mesmo do bruto)
- **Canais**: 4 (mesmo do bruto)
- **Processamento**: Filtro passa-banda 20-150 Hz

## Métricas Calculadas por Canal

1. **RMS** (Root Mean Square): Amplitude do sinal
2. **Freq. Mediana**: Frequência central da distribuição de potência
3. **ZCR** (Zero Crossing Rate): Taxa de cruzamento por zero
4. **Waveform Length**: Comprimento cumulativo da forma de onda

## Classificação de Modo

Baseada em percentis do RMS:
- **Baixa energia**: Q1 (0-33%)
- **Média energia**: Q2 (33-66%)
- **Alta energia**: Q3+ (66-100%)

## Troubleshooting

### "pylsl não instalado"
```bash
pip install pylsl
```

### "Nenhum stream LSL encontrado"
1. Verifique se o simulador está rodando
2. Certifique-se de que pylsl está instalado em ambos os terminais
3. Verifique se não há firewall bloqueando comunicação local

### "R_HOME não configurado"
1. Instale R (https://cran.r-project.org/)
2. Configure variável de ambiente `R_HOME` apontando para a pasta raiz do R
3. Instale rpy2: `pip install rpy2`

Se R não estiver disponível, o programa usa automaticamente fallback em matplotlib.

## Arquitetura

```
Grafico.py                  # Aplicação Principal (PyQt6)
├── JanelaNeuro              # Interface principal
├── calcular_parametros_sinal() # Cálculo de métricas
├── gerar_grafico_r()        # Geração de gráficos R
├── gerar_grafico_python()   # Fallback matplotlib
└── recarregar_de_lsl()      # Captura de LSL

simulador_lsl_emg.py        # Simulador de Streams LSL
├── SimuladorEMG             # Classe geradora
├── gerar_emg_sintetico()   # Síntese de sinal
├── filtrar_emg()            # Filtragem
└── enviar_stream()          # Broadcasting LSL
```

## Notas Importantes

- Os dados capturados são armazenados em `output/metricas_canais.csv`
- Os gráficos PNG são salvos em `output/` com formato de alta resolução (300 DPI)
- O web_view exibe HTML com as imagens dos gráficos
- A captura de LSL aguarda por 3 segundos antes de timeout
- Dados são capturados continuamente por 5 segundos quando "RECARREGAR" é clicado

## Versão
v1.2 Beta | 2026
