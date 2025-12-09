import os
from dotenv import load_dotenv
from groq import Groq

# 1. Cargar las variables de entorno
load_dotenv()

# 2. Obtener la clave
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("❌ ERROR: La variable GROQ_API_KEY no se encontró en .env")
else:
    try:
        # 3. Inicializar el cliente
        client = Groq(api_key=groq_api_key)

        # 4. Hacer una llamada de prueba
        print("✅ Probando la clave GROQ...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Solo di 'Hola'",
                }
            ],
           model="llama-3.1-8b-instant" # Un modelo rápido y accesible
        )
        
        # 5. Mostrar el resultado
        print("Respuesta de GROQ recibida:")
        print(f"   Mensaje: {chat_completion.choices[0].message.content.strip()}")
        print("\n🎉 La clave GROQ está activa y funcionando correctamente.")

    except Exception as e:
        print(f"\n❌ ERROR al usar la clave GROQ:")
        print(f"   Mensaje de error: {e}")
        print("   Posiblemente la clave es incorrecta o no tiene permisos.")