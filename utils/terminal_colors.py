#!/usr/bin/env python3
"""
SISTEMA DE CORES DO TERMINAL
============================

Módulo responsável por fornecer cores e formatação para o terminal.
"""

class TerminalColors:
    """Sistema de cores expandido para máxima visualização"""
    # Cores básicas
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Cores adicionais
    LIGHT_RED = '\033[101m'
    LIGHT_GREEN = '\033[102m'
    LIGHT_YELLOW = '\033[103m'
    LIGHT_BLUE = '\033[104m'
    LIGHT_MAGENTA = '\033[105m'
    LIGHT_CYAN = '\033[106m'
    
    # Estilos expandidos
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Backgrounds expandidos
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Reset
    RESET = '\033[0m'
    
    @staticmethod
    def colorize(text, color, style=None, bg=None):
        """Aplica cores e estilos ao texto"""
        result = ""
        if color:
            result += color
        if style:
            result += style
        if bg:
            result += bg
        result += str(text) + TerminalColors.RESET
        return result
    
    @staticmethod
    def success(text):
        """Texto de sucesso verde e negrito"""
        return TerminalColors.colorize(text, TerminalColors.GREEN, TerminalColors.BOLD)
    
    @staticmethod
    def error(text):
        """Texto de erro vermelho e negrito"""
        return TerminalColors.colorize(text, TerminalColors.RED, TerminalColors.BOLD)
    
    @staticmethod
    def warning(text):
        """Texto de aviso amarelo"""
        return TerminalColors.colorize(text, TerminalColors.YELLOW, TerminalColors.BOLD)
    
    @staticmethod
    def info(text):
        """Texto informativo azul"""
        return TerminalColors.colorize(text, TerminalColors.CYAN)
    
    @staticmethod
    def highlight(text):
        """Texto destacado com fundo"""
        return TerminalColors.colorize(text, TerminalColors.WHITE, TerminalColors.BOLD, TerminalColors.BG_BLUE) 