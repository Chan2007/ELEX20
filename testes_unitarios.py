"""
Testes unitários para validar componentes do sistema.
"""

import unittest
import numpy as np
import sys
from pathlib import Path


class TesteSimuladorEMG(unittest.TestCase):
    """Testes para o simulador de EMG."""

    def setUp(self):
        """Configuração antes de cada teste."""
        try:
            from simulador_lsl_emg import SimuladorEMG
            self.SimuladorEMG = SimuladorEMG
        except ImportError:
            self.skipTest("SimuladorEMG não pode ser importado")

    def teste_geracao_emg_sintetico(self):
        """Testa geração de sinal EMG sintético."""
        simulador = self.SimuladorEMG(sfreq=250, n_canais=4)
        tempo_s = 2
        emg = simulador.gerar_emg_sintetico(tempo_s)
        
        # Verificações
        self.assertEqual(emg.shape[0], 4, "Número de canais incorreto")
        self.assertEqual(emg.shape[1], int(250 * tempo_s), "Número de amostras incorreto")
        self.assertTrue(np.all(np.isfinite(emg)), "Contém NaN ou Inf")

    def teste_filtragem_emg(self):
        """Testa filtragem de sinal EMG."""
        simulador = self.SimuladorEMG(sfreq=250, n_canais=4)
        emg = simulador.gerar_emg_sintetico(2)
        emg_filt = simulador.filtrar_emg(emg)
        
        # Verificações
        self.assertEqual(emg_filt.shape, emg.shape, "Shape alterado após filtragem")
        self.assertTrue(np.all(np.isfinite(emg_filt)), "Contém NaN ou Inf após filtro")
        self.assertLess(np.max(np.abs(emg_filt)), np.max(np.abs(emg)) * 1.1, 
                       "Amplitude aumentou muito")


class TesteParametrosSinal(unittest.TestCase):
    """Testes para cálculo de parâmetros de sinal."""

    def teste_rms_sinal_puro(self):
        """Testa cálculo de RMS para sinal puro."""
        from Grafico import calcular_parametros_sinal
        
        # Criar sinal de teste (onda senoidal simples)
        t = np.linspace(0, 1, 1000)
        dados = np.sin(2 * np.pi * t)
        
        # Mock MNE Raw object
        class MockRaw:
            def __init__(self):
                self.info = {"sfreq": 1000}
                self.ch_names = ["Canal_1"]
                self._dados = dados.reshape(1, -1)
            
            def get_data(self):
                return self._dados
        
        raw = MockRaw()
        metricas, medias = calcular_parametros_sinal(raw)
        
        # RMS de sin(2π*t) deve ser aproximadamente 0.707
        rms = metricas[0]["rms"]
        self.assertAlmostEqual(rms, 1/np.sqrt(2), places=1, 
                              msg="RMS incorreto para sinal senoidal")

    def teste_zcr_sinal_oscilante(self):
        """Testa taxa de cruzamento por zero."""
        from Grafico import calcular_parametros_sinal
        
        # Sinal oscilante
        t = np.linspace(0, 10, 10000)
        dados = np.sin(2 * np.pi * t)  # 1 Hz
        
        class MockRaw:
            def __init__(self):
                self.info = {"sfreq": 1000}
                self.ch_names = ["Canal_1"]
                self._dados = dados.reshape(1, -1)
            
            def get_data(self):
                return self._dados
        
        raw = MockRaw()
        metricas, medias = calcular_parametros_sinal(raw)
        
        # ZCR deve ser > 0 para sinal oscilante
        zcr = metricas[0]["zcr"]
        self.assertGreater(zcr, 0, "ZCR deve ser positivo para sinal oscilante")

    def teste_waveform_length(self):
        """Testa cálculo de waveform length."""
        from Grafico import calcular_parametros_sinal
        
        # Sinal linear (incrementa uniformemente)
        dados = np.linspace(0, 10, 1000).reshape(1, -1)
        
        class MockRaw:
            def __init__(self):
                self.info = {"sfreq": 1000}
                self.ch_names = ["Canal_1"]
                self._dados = dados
            
            def get_data(self):
                return self._dados
        
        raw = MockRaw()
        metricas, medias = calcular_parametros_sinal(raw)
        
        # Waveform length deve ser aproximadamente 10
        wl = metricas[0]["waveform_length"]
        self.assertAlmostEqual(wl, 10, delta=1, msg="Waveform length incorreto")


class TesteInferirModo(unittest.TestCase):
    """Testes para inferência de modo de energia."""

    def teste_classificacao_modos(self):
        """Testa classificação em modos de energia."""
        from Grafico import inferir_modo_sinal
        
        metricas = [
            {"canal": "Ch1", "rms": 0.1, "freq_mediana": 50, "zcr": 0.1, "waveform_length": 5},
            {"canal": "Ch2", "rms": 0.5, "freq_mediana": 60, "zcr": 0.2, "waveform_length": 10},
            {"canal": "Ch3", "rms": 1.0, "freq_mediana": 70, "zcr": 0.3, "waveform_length": 15},
        ]
        
        linhas = inferir_modo_sinal(metricas)
        modos = [l["modo"] for l in linhas]
        
        # Verificar se há modos diferentes
        self.assertGreater(len(set(modos)), 1, "Deveria haver classificações diferentes")
        
        # Baixa energia deve ter RMS menor
        baixa_energia = [l for l in linhas if l["modo"] == "Baixa energia"]
        alta_energia = [l for l in linhas if l["modo"] == "Alta energia"]
        
        if baixa_energia and alta_energia:
            self.assertLess(baixa_energia[0]["rms"], alta_energia[0]["rms"],
                           "Baixa energia deve ter RMS < Alta energia")


class TesteLSL(unittest.TestCase):
    """Testes para funcionalidades de LSL."""

    def teste_import_pylsl(self):
        """Verifica se pylsl pode ser importado."""
        try:
            import pylsl
            self.assertTrue(True, "pylsl importado com sucesso")
        except ImportError:
            self.skipTest("pylsl não instalado - instale com: pip install pylsl")


def suite():
    """Cria suite de testes."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TesteSimuladorEMG))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TesteParametrosSinal))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TesteInferirModo))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TesteLSL))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
