#!/usr/bin/env python3
import os
import datetime
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519

def generate_jwt(private_key_path, key_id, override_algorithm=None):
    """
    Generates a JWT signed with the provided private key.
    
    This function auto-detects the key type:
      - For RSA keys, it uses RS256.
      - For Ed25519 keys, it rejects them with an error.
      
    Args:
        private_key_path (str): Path to the private key file.
        key_id (str): Unique identifier for the key (e.g. developer's email or key name).
        override_algorithm (str, optional): An optional override for the signing algorithm.
    
    Returns:
        str: The generated JWT.
        
    Raises:
        ValueError: If an Ed25519 key is used or if the override algorithm does not match the key type.
        RuntimeError: If key reading or JWT encoding fails.
    """
    try:
        with open(private_key_path, 'r') as f:
            ssh_key = f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading private key from {private_key_path}: {e}")

    try:
        key = serialization.load_ssh_private_key(ssh_key.encode(), password=None)
    except Exception as e:
        raise ValueError(f"Failed to load SSH private key: {e}")

    # Reject Ed25519 keys explicitly
    if isinstance(key, ed25519.Ed25519PrivateKey):
        raise ValueError("Ed25519 keys are not supported for JWT generation. Please use an RSA key instead.")
    elif not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("Unsupported key type. Only RSA keys are supported for JWT generation.")

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "iat": now,
        "exp": now + datetime.timedelta(minutes=5)
    }

    default_algorithm = "RS256"
    # Allow override of algorithm if provided (must match the key type)
    if override_algorithm:
        if override_algorithm != default_algorithm:
            raise ValueError(
                f"Provided algorithm '{override_algorithm}' does not match the key type. Expected '{default_algorithm}'."
            )
        algorithm = override_algorithm
    else:
        algorithm = default_algorithm

    headers = {
        "alg": algorithm,
        "typ": "JWT",
        "kid": key_id
    }

    try:
        token = jwt.encode(payload, key, algorithm=algorithm, headers=headers)
    except Exception as e:
        raise RuntimeError(f"JWT encoding failed: {e}")

    return token