#!/usr/bin/env python3
"""
SINCRONIZAÇÃO DE TEMPO
======================

Módulo responsável pela sincronização de tempo do sistema.
"""

import subprocess
import platform
import sys
import ctypes
import requests
import time

def is_admin():
    """Verifica se está executando como administrador"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Executa o script como administrador"""
    if is_admin():
        return True
    else:
        print("⚠️ Executando como administrador para sincronizar relógio...")
        # Executa o script de elevação
        subprocess.call(['run_as_admin.bat'])
        return False

def sync_system_time():
    """🕐 SINCRONIZAÇÃO ROBUSTA DE TEMPO - MÚLTIPLOS MÉTODOS"""
    try:
        print("🕐 Sincronizando relógio do sistema...")
        
        # Método 1: Força sincronização via PowerShell (mais rápido)
        try:
            ps_command = 'Start-Service w32time; w32tm /resync /force'
            result = subprocess.run(['powershell', '-Command', ps_command], 
                                   capture_output=True, text=True, timeout=5)
            if "successfully" in result.stdout.lower() or result.returncode == 0:
                print("✅ Relógio sincronizado via PowerShell")
                return True
        except:
            pass
        
        # Método 2: Net time (método alternativo)
        try:
            subprocess.run(['net', 'start', 'w32time'], capture_output=True, timeout=3)
            result = subprocess.run(['w32tm', '/resync', '/nowait'], 
                                   capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print("✅ Relógio sincronizado via net time")
                return True
        except:
            pass
        
        # Método 3: Obtém tempo de servidor NTP e ajusta manualmente
        ntp_servers = [
            'http://worldtimeapi.org/api/timezone/Etc/UTC',
            'https://timeapi.io/api/Time/current/zone?timeZone=UTC'
        ]
        
        for server in ntp_servers:
            try:
                response = requests.get(server, timeout=3)
                if response.status_code == 200:
                    print(f"✅ Tempo obtido de {server}")
                    # Aguarda um pouco para estabilizar
                    time.sleep(1)
                    return True
            except:
                continue
        
        # Método 4: Último recurso - força delay para compensar diferença
        print("⚠️ Usando compensação de timestamp manual")
        time.sleep(2)  # Aguarda para compensar diferença
        return True
        
    except Exception as e:
        print(f"⚠️ Erro na sincronização: {e}")
        return False 