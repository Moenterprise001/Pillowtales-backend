#!/usr/bin/env python3
"""
Generate Apple Client Secret JWT for Sign in with Apple

This script generates the JWT that Supabase requires for Apple OAuth.
The JWT is signed with ES256 using your .p8 private key.

Usage:
    python generate_apple_secret.py

You'll be prompted to enter:
    - Team ID (10-character Apple Developer Team ID)
    - Key ID (10-character Key ID from your .p8 file)
    - Client ID (your Service ID, e.g., com.pillowtales.login)
    - Private Key (paste your .p8 file contents)
"""

import jwt
import time
from datetime import datetime, timedelta

def generate_apple_client_secret(team_id: str, key_id: str, client_id: str, private_key: str, expiry_days: int = 180) -> str:
    """
    Generate an Apple client secret JWT.
    
    Args:
        team_id: Your Apple Developer Team ID (10 characters)
        key_id: The Key ID from your .p8 file (10 characters)
        client_id: Your Service ID (e.g., com.pillowtales.login)
        private_key: The contents of your .p8 private key file
        expiry_days: How many days until the JWT expires (max 180 = 6 months)
    
    Returns:
        The signed JWT string to use as Apple client secret
    """
    
    # Current time and expiry (max 6 months)
    now = int(time.time())
    expiry = now + (expiry_days * 24 * 60 * 60)  # Convert days to seconds
    
    # JWT headers
    headers = {
        "alg": "ES256",
        "kid": key_id
    }
    
    # JWT payload
    payload = {
        "iss": team_id,           # Issuer: Your Team ID
        "iat": now,               # Issued at: Current time
        "exp": expiry,            # Expiry: Within 6 months
        "aud": "https://appleid.apple.com",  # Audience: Apple
        "sub": client_id          # Subject: Your Service ID
    }
    
    # Sign the JWT with ES256 using the private key
    client_secret = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers=headers
    )
    
    return client_secret


def main():
    print("=" * 60)
    print("Apple Client Secret JWT Generator")
    print("=" * 60)
    print()
    
    # Get inputs
    print("Enter your Apple Developer credentials:")
    print()
    
    team_id = input("Team ID (10 characters): ").strip()
    if len(team_id) != 10:
        print(f"Warning: Team ID is usually 10 characters, got {len(team_id)}")
    
    key_id = input("Key ID (10 characters): ").strip()
    if len(key_id) != 10:
        print(f"Warning: Key ID is usually 10 characters, got {len(key_id)}")
    
    client_id = input("Client ID / Service ID [com.pillowtales.login]: ").strip()
    if not client_id:
        client_id = "com.pillowtales.login"
    
    print()
    print("Paste your .p8 private key (including -----BEGIN/END lines):")
    print("(Press Enter twice when done)")
    
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    # Remove trailing empty line
    if lines and lines[-1] == "":
        lines.pop()
    
    private_key = "\n".join(lines)
    
    if "-----BEGIN PRIVATE KEY-----" not in private_key:
        print("Error: Private key should start with '-----BEGIN PRIVATE KEY-----'")
        return
    
    print()
    print("Generating JWT...")
    print()
    
    try:
        client_secret = generate_apple_client_secret(
            team_id=team_id,
            key_id=key_id,
            client_id=client_id,
            private_key=private_key,
            expiry_days=180  # 6 months
        )
        
        # Calculate expiry date for display
        expiry_date = datetime.now() + timedelta(days=180)
        
        print("=" * 60)
        print("SUCCESS! Your Apple Client Secret JWT:")
        print("=" * 60)
        print()
        print(client_secret)
        print()
        print("=" * 60)
        print(f"This JWT expires on: {expiry_date.strftime('%Y-%m-%d')}")
        print()
        print("Next steps:")
        print("1. Copy the JWT above")
        print("2. Go to Supabase Dashboard → Authentication → Providers → Apple")
        print("3. Paste it into the 'Secret Key' field")
        print("4. Save")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error generating JWT: {e}")
        print()
        print("Make sure:")
        print("- Your .p8 key is valid")
        print("- PyJWT is installed with: pip install PyJWT")


if __name__ == "__main__":
    main()
