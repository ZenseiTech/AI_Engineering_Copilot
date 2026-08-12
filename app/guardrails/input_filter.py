import re
from fastapi import HTTPException, status

# Patterns for catching sensitive internal keys and prompt injection attempts
SECRET_PATTERNS = [
    (re.compile(r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*"), "[REDACTED_JWT]"),
    (re.compile(r"sk-[a-zA-Z0-9]{32,64}"), "[REDACTED_API_KEY]"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_TOKEN]"),
]

INJECTION_PATTERNS = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"system override", re.IGNORECASE),
    re.compile(r"you are now an unrestricted ai", re.IGNORECASE),
]

def sanitize_dev_prompt(prompt: str) -> str:
    """
    Checks for prompt injection attacks and redacts internal credentials.
    """
    # 1. Reject Prompt Injections
    for pattern in INJECTION_PATTERNS:
        if pattern.search(prompt):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security policy violation: Injection pattern detected in prompt."
            )
            
    # 2. Redact Secrets inline
    sanitized = prompt
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
        
    return sanitized