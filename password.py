import bcrypt

import secrets

cookie_name = secrets.token_hex(8)  # Génère un nom de cookie aléatoire
print("Cookie Name:", cookie_name)


