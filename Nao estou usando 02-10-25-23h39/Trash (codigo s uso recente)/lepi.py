# test_login_usuario.py
from db import Session
from sqlalchemy import text
from passlib.hash import pbkdf2_sha256

# Usuário e senha que quer testar
usuario_input = "Lepi"
senha_input = "faixab123"

# Cria sessão do banco
session = Session()

try:
    # Busca hash da senha e forcar_reset do usuário
    result = session.execute(
        text("SELECT senha, forcar_reset FROM usuarios WHERE usuario = :usuario"),
        {"usuario": usuario_input}
    ).fetchone()

    if result:
        hash_senha, forcar_reset = result

        # Verifica senha
        if pbkdf2_sha256.verify(senha_input, hash_senha):
            print("Login válido ✅")
            print(f"forcar_reset = {forcar_reset}")
        else:
            print("Senha inválida ❌")
    else:
        print("Usuário não encontrado ❌")

except Exception as e:
    print("Erro ao consultar o banco:", e)

finally:
    session.close()
