from fastapi.security import APIKeyHeader
from fastapi import HTTPException, status, Security

# API Keys - this would in a real app be stored elsewhere, not hardcoded (example only)
api_keys = ['i+3dDJKely1Z5H8C7+C0wQ==']
security_scheme = APIKeyHeader(name="x-api-key")

def auth_api_key(input_api_key_string: str = Security(security_scheme)):
    if input_api_key_string not in api_keys:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            'Invalid authentication credentials'
        )