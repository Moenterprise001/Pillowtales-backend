#!/usr/bin/env python3
"""
Quick Apple Client Secret JWT Generator

Run with:
    python quick_apple_jwt.py <team_id> <key_id> <path_to_p8_file>

Example:
    python quick_apple_jwt.py ABC123DEF0 XYZ987WVU1 /path/to/AuthKey.p8
"""

import sys
import jwt
import time
from datetime import datetime, timedelta

def generate_jwt(team_id: str, key_id: str, private_key: str) -> str:
    now = int(time.time())
    expiry = now + (180 * 24 * 60 * 60)  # 6 months
    
    headers = {"alg": "ES256", "kid": key_id}
    payload = {
        "iss": team_id,
        "iat": now,
        "exp": expiry,
        "aud": "https://appleid.apple.com",
        "sub": "com.pillowtales.login"
    }
    
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python quick_apple_jwt.py <team_id> <key_id> <p8_file_path>")
        print("Example: python quick_apple_jwt.py ABC123DEF0 XYZ987WVU1 ./AuthKey.p8")
        sys.exit(1)
    
    team_id = sys.argv[1]
    key_id = sys.argv[2]
    p8_path = sys.argv[3]
    
    with open(p8_path, 'r') as f:
        private_key = f.read()
    
    jwt_secret = generate_jwt(team_id, key_id, private_key)
    
    expiry_date = datetime.now() + timedelta(days=180)
    print(f"\n{'='*60}")
    print("Apple Client Secret JWT (expires {})".format(expiry_date.strftime('%Y-%m-%d')))
    print(f"{'='*60}\n")
    print(jwt_secret)
    print(f"\n{'='*60}")
    print("Paste this into Supabase → Auth → Providers → Apple → Secret Key")
    print(f"{'='*60}\n")
