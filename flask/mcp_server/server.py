#!/usr/bin/env python3
import asyncio
import json
import sys
from datetime import datetime
import math
import random

# ============ HERRAMIENTA: CALCULADORA ============
def calculator(operation, a, b):
    """Calculadora simple con operaciones básicas"""
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else "Error: División por cero",
        'sqrt': lambda x, _: math.sqrt(x),
    }

    if operation in operations:
        return operations[operation](a, b)
    return f"Operación '{operation}' no soportada"


# ============ HERRAMIENTA: CLIMA ============
def weather_tool(city):
    """Simulador de clima (en producción usarías una API real)"""
    # Datos simulados para demostración
    cities_data = {
        'madrid': {'temp': 22, 'condition': 'Soleado', 'humidity': 45},
        'barcelona': {'temp': 24, 'condition': 'Parcialmente nublado', 'humidity': 65},
        'mexico': {'temp': 28, 'condition': 'Lluvioso', 'humidity': 80},
        'new york': {'temp': 18, 'condition': 'Nublado', 'humidity': 70},
        'tokyo': {'temp': 20, 'condition': 'Despejado', 'humidity': 55}
    }

    city_lower = city.lower()

    # Si la ciudad está en nuestros datos, usarla
    if city_lower in cities_data:
        data = cities_data[city_lower]
    else:
        # Generar datos aleatorios para ciudades desconocidas
        data = {
            'temp': random.randint(15, 35),
            'condition': random.choice(['Soleado', 'Nublado', 'Lluvioso', 'Despejado']),
            'humidity': random.randint(40, 90)
        }

    return {
        'city': city,
        'temperature': f"{data['temp']}°C",
        'condition': data['condition'],
        'humidity': f"{data['humidity']}%",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ============ HERRAMIENTA: PROCESADOR DE TEXTO ============
def text_processor(text, operation):
    """Procesador de texto con varias operaciones"""
    operations = {
        'uppercase': lambda t: t.upper(),
        'lowercase': lambda t: t.lower(),
        'reverse': lambda t: t[::-1],
        'count_words': lambda t: f"{len(t.split())} palabras",
        'count_chars': lambda t: f"{len(t)} caracteres"
    }

    if operation in operations:
        return {
            'original': text,
            'result': operations[operation](text),
            'operation': operation
        }

    return f"Operación '{operation}' no soportada"

# ============ HERRAMIENTA: CALCULADORA ============
def converter(value, from_unit, to_unit):
    """Convertidor de unidades mejorado con más conversiones y mejor manejo de errores"""
    try:
        # Validar que el valor sea numérico
        numeric_value = float(value)
    except (ValueError, TypeError):
        return {"error": f"Valor '{value}' no es numérico"}
    
    # Definir todas las conversiones soportadas
    conversions = {
        # Distancia
        'km': {
            'mi': lambda x: x * 0.621371,
            'm': lambda x: x * 1000,
            'cm': lambda x: x * 100000,
            'ft': lambda x: x * 3280.84,
            'in': lambda x: x * 39370.1
        },
        'mi': {
            'km': lambda x: x * 1.60934,
            'm': lambda x: x * 1609.34,
            'cm': lambda x: x * 160934,
            'ft': lambda x: x * 5280,
            'in': lambda x: x * 63360
        },
        'm': {
            'km': lambda x: x / 1000,
            'mi': lambda x: x / 1609.34,
            'cm': lambda x: x * 100,
            'ft': lambda x: x * 3.28084,
            'in': lambda x: x * 39.3701
        },
        'cm': {
            'km': lambda x: x / 100000,
            'mi': lambda x: x / 160934,
            'm': lambda x: x / 100,
            'ft': lambda x: x / 30.48,
            'in': lambda x: x / 2.54
        },
        'ft': {
            'km': lambda x: x / 3280.84,
            'mi': lambda x: x / 5280,
            'm': lambda x: x / 3.28084,
            'cm': lambda x: x * 30.48,
            'in': lambda x: x * 12
        },
        'in': {
            'km': lambda x: x / 39370.1,
            'mi': lambda x: x / 63360,
            'm': lambda x: x / 39.3701,
            'cm': lambda x: x * 2.54,
            'ft': lambda x: x / 12
        },
        
        # Temperatura
        'C': {
            'F': lambda x: (x * 9 / 5) + 32,
            'K': lambda x: x + 273.15
        },
        'F': {
            'C': lambda x: (x - 32) * 5 / 9,
            'K': lambda x: ((x - 32) * 5 / 9) + 273.15
        },
        'K': {
            'C': lambda x: x - 273.15,
            'F': lambda x: ((x - 273.15) * 9 / 5) + 32
        },
        
        # Peso
        'kg': {
            'lb': lambda x: x * 2.20462,
            'g': lambda x: x * 1000,
            'oz': lambda x: x * 35.274
        },
        'lb': {
            'kg': lambda x: x / 2.20462,
            'g': lambda x: x * 453.592,
            'oz': lambda x: x * 16
        },
        'g': {
            'kg': lambda x: x / 1000,
            'lb': lambda x: x / 453.592,
            'oz': lambda x: x / 28.3495
        },
        'oz': {
            'kg': lambda x: x / 35.274,
            'lb': lambda x: x / 16,
            'g': lambda x: x * 28.3495
        }
    }
    
    # Verificar si la conversión existe
    if from_unit not in conversions:
        available_units = list(conversions.keys())
        return {"error": f"Unidad '{from_unit}' no soportada. Unidades disponibles: {available_units}"}
    
    if to_unit not in conversions[from_unit]:
        available_conversions = list(conversions[from_unit].keys())
        return {"error": f"Conversión de '{from_unit}' a '{to_unit}' no soportada. Conversiones disponibles desde '{from_unit}': {available_conversions}"}
    
    # Realizar la conversión
    try:
        result = conversions[from_unit][to_unit](numeric_value)
        return {
            "original_value": numeric_value,
            "original_unit": from_unit,
            "converted_value": round(result, 6),
            "converted_unit": to_unit,
            "conversion": f"{numeric_value} {from_unit} = {round(result, 6)} {to_unit}"
        }
    except Exception as e:
        return {"error": f"Error en la conversión: {str(e)}"}

# ============ PROTOCOLO MCP ============
class MCPServer:
    def __init__(self):
        self.request_id = 0
    
    def get_tools(self):
        """Lista todas las herramientas disponibles"""
        return [
            {
                "name": "calculator",
                "description": "Realiza operaciones matemáticas básicas",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            },
            {
                "name": "weather",
                "description": "Obtiene información del clima de una ciudad",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "text_processor",
                "description": "Procesa texto con varias operaciones",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "operation": {"type": "string", "enum": ["uppercase", "lowercase", "reverse", "count_words", "count_chars"]}
                    },
                    "required": ["text", "operation"]
                }
            },
            {
                "name": "converter",
                "description": "Convertidor de unidades mejorado. Soporta conversiones de distancia (km, mi, m, cm, ft, in), temperatura (C, F, K) y peso (kg, lb, g, oz)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "from_unit": {"type": "string"},
                        "to_unit": {"type": "string"}
                    },
                    "required": ["value", "from_unit", "to_unit"]
                }
            }
        ]
    
    def call_tool(self, name, arguments):
        """Ejecuta una herramienta específica"""
        try:
            if name == "calculator":
                result = calculator(
                    arguments.get("operation"),
                    arguments.get("a", 0),
                    arguments.get("b", 0)
                )
                return [{"type": "text", "text": str(result)}]
            
            elif name == "weather":
                result = weather_tool(arguments.get("city", "Madrid"))
                return [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
            
            elif name == "text_processor":
                result = text_processor(
                    arguments.get("text", ""),
                    arguments.get("operation", "uppercase")
                )
                return [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
            
            elif name == "converter":
                result = converter(
                    arguments.get("value", 0),
                    arguments.get("from_unit", "km"),
                    arguments.get("to_unit", "mi")
                )
                return [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
            
            else:
                return [{"type": "text", "text": f"Error: Herramienta '{name}' no encontrada"}]
        
        except Exception as e:
            return [{"type": "text", "text": f"Error: {str(e)}"}]
    
    async def handle_message(self, message):
        """Maneja mensajes del protocolo MCP"""
        try:
            # Asegurar que el mensaje tenga un ID válido
            msg_id = message.get("id", 0)
            if msg_id is None:
                msg_id = 0
            
            if message["method"] == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "flask-tools-server",
                            "version": "1.0.0"
                        }
                    }
                }
            
            elif message["method"] == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": self.get_tools()
                    }
                }
            
            elif message["method"] == "tools/call":
                name = message["params"]["name"]
                arguments = message["params"]["arguments"]
                content = self.call_tool(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": content
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {message.get('method')}"
                    }
                }
        
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id", 0) if message.get("id") is not None else 0,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }


def main():
    """Función principal del servidor MCP"""
    server = MCPServer()
    
    print("🚀 Servidor MCP iniciando...", file=sys.stderr)
    print("📊 Herramientas disponibles: calculator, weather, text_processor, converter", file=sys.stderr)
    
    # Leer mensajes desde stdin y escribir respuestas a stdout
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue
                
            # Parsear mensaje JSON
            message = json.loads(line)
            
            # Procesar mensaje (convertir a sync)
            response = asyncio.run(server.handle_message(message))
            
            # Enviar respuesta a stdout
            print(json.dumps(response, ensure_ascii=False), flush=True)
            
        except json.JSONDecodeError as e:
            print(f"Error JSON: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            # Error interno
            error_response = {
                "jsonrpc": "2.0",
                "id": 0,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()