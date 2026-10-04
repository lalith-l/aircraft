"""
Nirnay Store — Cryptographic utilities and Node Registry.

Status: [REAL]
"""

import nacl.signing
import nacl.encoding
import nacl.exceptions
from typing import Dict, Tuple

class NodeIdentity:
    """Manages the identity (keys) for a node."""
    def __init__(self, node_id: str, seed: bytes = None):
        self.node_id = node_id
        if seed:
            self.signing_key = nacl.signing.SigningKey(seed)
        else:
            self.signing_key = nacl.signing.SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

    def public_key_hex(self) -> str:
        """Return the public key in hex format."""
        return self.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode('utf-8')

    def sign(self, message: str) -> str:
        """Sign a message and return the hex signature."""
        signed = self.signing_key.sign(message.encode('utf-8'))
        return signed.signature.hex()


class KeyRegistry:
    """A registry of known public keys for nodes."""
    def __init__(self):
        self._keys: Dict[str, nacl.signing.VerifyKey] = {}

    def register(self, node_id: str, public_key_hex: str):
        """Register a node's public key."""
        vk_bytes = bytes.fromhex(public_key_hex)
        self._keys[node_id] = nacl.signing.VerifyKey(vk_bytes)

    def is_known(self, node_id: str) -> bool:
        """Check if a node is registered."""
        return node_id in self._keys

    def verify(self, node_id: str, message: str, signature_hex: str) -> bool:
        """Verify a signature against the registered public key."""
        if node_id not in self._keys:
            return False
        
        vk = self._keys[node_id]
        try:
            vk.verify(message.encode('utf-8'), bytes.fromhex(signature_hex))
            return True
        except nacl.exceptions.BadSignatureError:
            return False

# Global registry for convenience in the prototype
global_registry = KeyRegistry()
