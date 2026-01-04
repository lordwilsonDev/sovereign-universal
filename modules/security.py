#!/usr/bin/env python3
"""
🛡️ SOVEREIGN SECURITY LAYER
=============================
Advanced security hardening for the Sovereign Controller.

Features:
- Request timeouts with async execution
- Rate limiting per client/IP
- Input sanitization (strip nulls, normalize unicode)
- Memory file size limits
- Circuit breaker pattern
- Audit logging
"""

import time
import threading
import functools
import hashlib
import unicodedata
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from collections import defaultdict
from datetime import datetime, timedelta


# ============================================================================
# ⏱️ TIMEOUT DECORATOR
# ============================================================================

class TimeoutError(Exception):
    """Raised when a function times out"""
    pass


def with_timeout(seconds: float = 30.0):
    """Decorator to add timeout to any function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator


# ============================================================================
# 🚦 RATE LIMITER
# ============================================================================

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Max requests in 1 second


class RateLimiter:
    """Token bucket rate limiter with sliding window"""
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
    
    def _cleanup_old(self, client_id: str, window_seconds: int):
        """Remove requests older than window"""
        now = time.time()
        cutoff = now - window_seconds
        self._requests[client_id] = [t for t in self._requests[client_id] if t > cutoff]
    
    def check(self, client_id: str = "default") -> tuple[bool, str]:
        """Check if request is allowed. Returns (allowed, reason)"""
        with self._lock:
            now = time.time()
            
            # Cleanup old requests
            self._cleanup_old(client_id, 3600)  # 1 hour window
            
            requests = self._requests[client_id]
            
            # Check burst (last 1 second)
            recent = sum(1 for t in requests if t > now - 1)
            if recent >= self.config.burst_limit:
                return False, f"Burst limit exceeded ({self.config.burst_limit}/sec)"
            
            # Check per-minute limit
            last_minute = sum(1 for t in requests if t > now - 60)
            if last_minute >= self.config.requests_per_minute:
                return False, f"Rate limit exceeded ({self.config.requests_per_minute}/min)"
            
            # Check per-hour limit
            if len(requests) >= self.config.requests_per_hour:
                return False, f"Hourly limit exceeded ({self.config.requests_per_hour}/hour)"
            
            # Allow and record
            self._requests[client_id].append(now)
            return True, "OK"
    
    def get_stats(self, client_id: str = "default") -> dict:
        """Get rate limit stats for a client"""
        with self._lock:
            self._cleanup_old(client_id, 3600)
            requests = self._requests[client_id]
            now = time.time()
            return {
                "last_minute": sum(1 for t in requests if t > now - 60),
                "last_hour": len(requests),
                "remaining_minute": max(0, self.config.requests_per_minute - sum(1 for t in requests if t > now - 60)),
                "remaining_hour": max(0, self.config.requests_per_hour - len(requests))
            }


# ============================================================================
# 🧹 INPUT SANITIZATION
# ============================================================================

class InputSanitizer:
    """Sanitize and normalize user inputs"""
    
    # Maximum lengths
    MAX_QUERY_LENGTH = 100_000  # 100KB
    MAX_TOOL_ARG_LENGTH = 10_000  # 10KB per arg
    
    # Dangerous patterns to remove
    STRIP_PATTERNS = [
        r'\x00',  # Null bytes
        r'\x1b\[[0-9;]*m',  # ANSI escape codes
        r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # Control characters (except newline, tab, CR)
    ]
    
    # Unicode normalization
    NORMALIZE_FORM = 'NFKC'  # Compatibility decomposition + canonical composition
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """Sanitize a user query"""
        if not isinstance(query, str):
            query = str(query)
        
        # Truncate
        if len(query) > cls.MAX_QUERY_LENGTH:
            query = query[:cls.MAX_QUERY_LENGTH]
        
        # Normalize unicode (prevents lookalike attacks)
        query = unicodedata.normalize(cls.NORMALIZE_FORM, query)
        
        # Strip dangerous patterns
        for pattern in cls.STRIP_PATTERNS:
            query = re.sub(pattern, '', query)
        
        return query
    
    @classmethod
    def sanitize_tool_args(cls, args: dict) -> dict:
        """Sanitize tool arguments"""
        sanitized = {}
        for key, value in args.items():
            # Sanitize key
            clean_key = cls.sanitize_query(str(key))[:256]
            
            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = cls.sanitize_query(value)
                if len(clean_value) > cls.MAX_TOOL_ARG_LENGTH:
                    clean_value = clean_value[:cls.MAX_TOOL_ARG_LENGTH]
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            elif isinstance(value, dict):
                clean_value = cls.sanitize_tool_args(value)
            elif isinstance(value, list):
                clean_value = [cls.sanitize_query(str(v))[:cls.MAX_TOOL_ARG_LENGTH] for v in value[:100]]
            else:
                clean_value = str(value)[:cls.MAX_TOOL_ARG_LENGTH]
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    @classmethod
    def is_suspicious(cls, text: str) -> tuple[bool, str]:
        """Check if text looks suspicious"""
        reasons = []
        
        # Check for high special character ratio
        if len(text) > 0:
            special = sum(1 for c in text if not c.isalnum() and not c.isspace())
            if special / len(text) > 0.5:
                reasons.append("High special character ratio")
        
        # Check for repeated characters (possible DoS)
        if len(text) > 100:
            char_counts = {}
            for c in text:
                char_counts[c] = char_counts.get(c, 0) + 1
            max_ratio = max(char_counts.values()) / len(text)
            if max_ratio > 0.8:
                reasons.append("Repetitive content")
        
        # Check for encoded content
        if '%' in text and len(re.findall(r'%[0-9a-fA-F]{2}', text)) > 5:
            reasons.append("Possible URL encoding")
        
        return bool(reasons), "; ".join(reasons)


# ============================================================================
# 📊 AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._log: list = []
        self._lock = threading.Lock()
    
    def log(self, event_type: str, details: dict, severity: str = "INFO"):
        """Log a security event"""
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "severity": severity,
                "details": details
            }
            self._log.append(entry)
            
            # Trim if too large
            if len(self._log) > self.max_entries:
                self._log = self._log[-self.max_entries:]
            
            # Print warnings and above
            if severity in ("WARNING", "ERROR", "CRITICAL"):
                print(f"🛡️ [{severity}] {event_type}: {details}")
    
    def get_recent(self, count: int = 100, severity: str = None) -> list:
        """Get recent audit entries"""
        with self._lock:
            entries = self._log[-count:]
            if severity:
                entries = [e for e in entries if e["severity"] == severity]
            return entries
    
    def get_stats(self) -> dict:
        """Get audit statistics"""
        with self._lock:
            by_type = {}
            by_severity = {}
            for entry in self._log:
                by_type[entry["type"]] = by_type.get(entry["type"], 0) + 1
                by_severity[entry["severity"]] = by_severity.get(entry["severity"], 0) + 1
            return {
                "total_entries": len(self._log),
                "by_type": by_type,
                "by_severity": by_severity
            }


# ============================================================================
# 🔌 CIRCUIT BREAKER
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying again
    half_open_max: int = 3  # Test requests in half-open state


class CircuitBreaker:
    """Circuit breaker pattern for external services"""
    
    STATE_CLOSED = "closed"  # Normal operation
    STATE_OPEN = "open"  # Rejecting requests
    STATE_HALF_OPEN = "half_open"  # Testing recovery
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = self.STATE_CLOSED
        self._failures = 0
        self._last_failure = 0
        self._half_open_attempts = 0
        self._lock = threading.Lock()
    
    @property
    def state(self):
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == self.STATE_OPEN:
                if time.time() - self._last_failure > self.config.recovery_timeout:
                    self._state = self.STATE_HALF_OPEN
                    self._half_open_attempts = 0
            return self._state
    
    def allow_request(self) -> bool:
        """Check if request should be allowed"""
        state = self.state
        if state == self.STATE_CLOSED:
            return True
        elif state == self.STATE_OPEN:
            return False
        else:  # HALF_OPEN
            with self._lock:
                if self._half_open_attempts < self.config.half_open_max:
                    self._half_open_attempts += 1
                    return True
                return False
    
    def record_success(self):
        """Record a successful request"""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._state = self.STATE_CLOSED
            self._failures = 0
    
    def record_failure(self):
        """Record a failed request"""
        with self._lock:
            self._failures += 1
            self._last_failure = time.time()
            
            if self._state == self.STATE_HALF_OPEN:
                self._state = self.STATE_OPEN
            elif self._failures >= self.config.failure_threshold:
                self._state = self.STATE_OPEN


# ============================================================================
# 🛡️ SECURITY LAYER (Combines All)
# ============================================================================

class SecurityLayer:
    """Main security layer combining all hardening features"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.sanitizer = InputSanitizer()
        self.audit = AuditLogger()
        self.circuits: Dict[str, CircuitBreaker] = {}
        self._enabled = True
    
    def enable(self):
        self._enabled = True
        self.audit.log("SECURITY_ENABLED", {})
    
    def disable(self):
        self._enabled = False
        self.audit.log("SECURITY_DISABLED", {}, severity="WARNING")
    
    def get_circuit(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        if name not in self.circuits:
            self.circuits[name] = CircuitBreaker(name)
        return self.circuits[name]
    
    def check_request(self, query: str, client_id: str = "default") -> tuple[bool, str, str]:
        """
        Full security check on a request.
        Returns: (allowed, sanitized_query, reason)
        """
        if not self._enabled:
            return True, query, "Security disabled"
        
        # Rate limit check
        allowed, reason = self.rate_limiter.check(client_id)
        if not allowed:
            self.audit.log("RATE_LIMITED", {"client": client_id, "reason": reason}, "WARNING")
            return False, query, reason
        
        # Sanitize input
        sanitized = self.sanitizer.sanitize_query(query)
        
        # Suspicious content check
        suspicious, sus_reason = self.sanitizer.is_suspicious(sanitized)
        if suspicious:
            self.audit.log("SUSPICIOUS_INPUT", {
                "client": client_id,
                "reason": sus_reason,
                "length": len(query)
            }, "WARNING")
        
        # Log the request
        self.audit.log("REQUEST", {
            "client": client_id,
            "length": len(sanitized),
            "suspicious": suspicious
        })
        
        return True, sanitized, "OK"
    
    def wrap_with_timeout(self, func: Callable, timeout: float = 30.0) -> Callable:
        """Wrap a function with timeout"""
        return with_timeout(timeout)(func)


# Global security layer instance
security = SecurityLayer()


# ============================================================================
# 🧪 TEST
# ============================================================================

if __name__ == "__main__":
    print("🛡️ Security Layer Tests\n")
    
    # Test rate limiter
    print("1. Rate Limiter:")
    rl = RateLimiter(RateLimitConfig(requests_per_minute=5, burst_limit=2))
    for i in range(7):
        allowed, reason = rl.check("test")
        print(f"   Request {i+1}: {'✅' if allowed else '❌'} {reason}")
    
    # Test sanitizer
    print("\n2. Input Sanitizer:")
    evil = "Hello\x00World\x1b[31mRed\x1b[0m"
    clean = InputSanitizer.sanitize_query(evil)
    print(f"   Original: {repr(evil)}")
    print(f"   Sanitized: {repr(clean)}")
    
    # Test timeout
    print("\n3. Timeout:")
    @with_timeout(0.5)
    def slow_func():
        time.sleep(2)
        return "Done"
    
    try:
        slow_func()
        print("   ❌ Should have timed out")
    except TimeoutError as e:
        print(f"   ✅ {e}")
    
    # Test circuit breaker
    print("\n4. Circuit Breaker:")
    cb = CircuitBreaker("test_service")
    for i in range(7):
        if cb.allow_request():
            cb.record_failure()
            print(f"   Request {i+1}: Failed, state={cb.state}")
        else:
            print(f"   Request {i+1}: ❌ Blocked by circuit breaker")
    
    print("\n✅ All security tests complete!")
