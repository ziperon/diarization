from cryptography.fernet import Fernet
import base64
import os
from typing import Optional
import argparse

# Путь к файлу с ключом (хранится в секрете)
KEY_FILE = "secret.key"

def generate_key() -> bytes:
    """
    Генерирует новый 32-байтовый ключ (base64) и сохраняет его в файл.
    Вызывается один раз при настройке системы.
    """
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)
    print(f"🔐 Ключ сгенерирован и сохранён в {KEY_FILE}")
    return key

def load_key():
    if not os.path.exists(KEY_FILE):
        print("⚠️ Файл ключа не найден. Генерирую новый...")
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        print(f"🔐 Ключ сгенерирован и сохранён в {KEY_FILE}")
        return key
    else:
        with open(KEY_FILE, "rb") as f:
            return f.read()

def encrypt_password(password: str, key: Optional[bytes] = None) -> str:
    """
    Шифрует пароль с использованием Fernet.
    :param password: Пароль для шифрования
    :param key: Ключ (если не передан — загружается из файла)
    :return: Зашифрованный пароль в base64
    """
    if key is None:
        key = load_key()
    
    fernet = Fernet(key)
    encrypted_password = fernet.encrypt(password.encode())
    
    return encrypted_password.decode()

def decrypt_password(encrypted_password: str, key: Optional[bytes] = None) -> str:
    if key is None:
        key = load_key()

    fernet = Fernet(key)

    try:
        encrypted_bytes = str(encrypted_password).encode('utf-8')
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')

    except Exception as e:
        raise ValueError(f"Ошибка дешифрования: неверный ключ или повреждённые данные {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Шифрование и дешифрование паролей с помощью Fernet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
python crypto.py encrypt "my_password"
python crypto.py decrypt "gAAAAABl..."
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="Команда")

    # Команда: encrypt
    encrypt_parser = subparsers.add_parser("encrypt", help="Зашифровать пароль")
    encrypt_parser.add_argument("password", type=str, help="Пароль для шифрования")

    # Команда: decrypt
    decrypt_parser = subparsers.add_parser("decrypt", help="Расшифровать пароль")
    decrypt_parser.add_argument("encrypted", type=str, help="Зашифрованный пароль (base64)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        key = load_key()

        if args.command == "encrypt":
            encrypted = encrypt_password(args.password, key)
            print(f"🔐 Зашифрованный пароль: {encrypted}")
            # Дополнительно: можно скопировать в буфер обмена (если нужно)
            # import pyperclip
            # pyperclip.copy(encrypted)
            # print("✅ Скопировано в буфер обмена.")

        elif args.command == "decrypt":
            decrypted = decrypt_password(args.encrypted, key)
            print(f"🔓 Расшифрованный пароль: {decrypted}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()