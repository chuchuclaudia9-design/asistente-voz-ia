import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository

# 1. Cargar las variables de entorno
load_dotenv()

# 2. Obtener la clave
hf_api_key = os.getenv("HF_API_KEY")

if not hf_api_key:
    print("❌ ERROR: La variable HF_API_KEY no se encontró en .env")
else:
    try:
        # 3. Inicializar el cliente (la clave se usa internamente)
        api = HfApi(token=hf_api_key)

        # 4. Hacer una llamada de prueba (listar uno de tus repositorios)
        # Esto verifica que el token de lectura ('Read') funciona.
        print("✅ Probando la clave Hugging Face...")
        
        # Intentamos obtener la información de tu perfil usando el token
        user_info = api.whoami(token=hf_api_key) 
        
        print("Información de usuario obtenida:")
        print(f"   Usuario (sub): {user_info['name']}")

        # Opcional: Probar a listar un repositorio que *deberías* poder leer
        # Reemplaza 'google/flan-t5-small' por cualquier modelo público
        model_info = api.model_info("google/flan-t5-small")
        print(f"   Acceso exitoso al modelo: {model_info.modelId}")
        
        print("\n🎉 La clave Hugging Face está activa y funcionando correctamente.")

    except Exception as e:
        print(f"\n❌ ERROR al usar la clave Hugging Face:")
        print(f"   Mensaje de error: {e}")
        print("   Posiblemente la clave es incorrecta o no tiene permiso de 'Read'.")