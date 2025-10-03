from passlib.hash import pbkdf2_sha256

# Hash que você gerou
hash_senha = "$pbkdf2-sha256$29000$K4VwrpXSWouRcu5di/He2w$DFzMSUxm.Xblo63jU01iYggpIkN43pQf3FaH4LAXxiI"

# Senha que o usuário vai digitar
senha_input = "faixab123"

# Verifica se bate
if pbkdf2_sha256.verify(senha_input, hash_senha):
    print("Senha correta ✅")
else:
    print("Senha incorreta ❌")
