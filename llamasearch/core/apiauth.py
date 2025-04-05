#!/usr/bin/env python3
import jwt
import os
import sys
import argparse
import datetime

def load_private_key(private_key_path):
    """Loads the private key from the specified file."""
    try:
        with open(private_key_path, "r") as f:
            key = f.read()
        return key
    except Exception as e:
        raise RuntimeError(f"Error reading private key from {private_key_path}: {e}")

def load_public_key(private_key_path):
    """Assumes the public key is in the same directory with a .pub extension."""
    public_key_path = private_key_path + ".pub"
    if not os.path.exists(public_key_path):
        raise RuntimeError(f"Public key file not found: {public_key_path}")
    try:
        with open(public_key_path, "r") as f:
            pub_key = f.read().strip()
        return pub_key
    except Exception as e:
        raise RuntimeError(f"Error reading public key from {public_key_path}: {e}")

def generate_jwt(private_key_path, algorithm="RS256"):
    """
    Generates a JWT signed with the private key.
    
    The JWT includes:
      - Issued at (iat) and expiration (exp) claims.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
    
    # Load the SSH private key
    with open(private_key_path, 'r') as f:
        ssh_key = f.read()
    
    # Convert SSH key to appropriate format
    key = serialization.load_ssh_private_key(ssh_key.encode(), password=None)
    
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "iat": now,
        "exp": now + datetime.timedelta(minutes=5)
    }
    
    # Handle key format based on algorithm
    if algorithm.startswith('RS'):
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("RSA algorithm specified but key is not an RSA key")
        # For RSA, convert to PEM format
        private_key = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
    else:  # For Ed25519
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise ValueError("EdDSA algorithm expected but key is not an Ed25519 key")
        # For Ed25519, convert to PEM format
        private_key = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        algorithm = 'EdDSA'  # Force EdDSA algorithm for Ed25519 keys
    
    token = jwt.encode(payload, private_key, algorithm=algorithm)
    return token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JWT using a private key from your $HOME/.ssh directory")
    parser.add_argument("--key", help="Path to the private key file (default: ~/.ssh/id_ed25519)", default=os.path.expanduser("~/.ssh/id_ed25519"))
    parser.add_argument("--alg", help="Signing algorithm (e.g. EdDSA for Ed25519, RS256 for RSA)", default="EdDSA")
    args = parser.parse_args()
    try:
        token = generate_jwt(args.key, args.alg)
        print("Generated JWT:")
        print(token)
    except Exception as e:
        print("Error generating JWT:", str(e))
        sys.exit(1)