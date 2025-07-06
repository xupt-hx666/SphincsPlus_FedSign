import torch
from args import args_parser
from sphincs import SPHINCSPlus
import time

args = args_parser()


class SphincsCPU:
    def __init__(self, security_level=128):
        self.sphincs = SPHINCSPlus(security_level)
        self.keygen_time_ms = None
        self._generate_keys()

    def _generate_keys(self):
        start_time = time.time()
        self.public_key, self.private_key = self.sphincs.keygen()
        self.keygen_time_ms = (time.time() - start_time) * 1000
        print(f"SPHINCS+密钥生成时间: {self.keygen_time_ms:.2f}ms")

    def sign(self, data: bytes) -> tuple:
        start_time = time.time()
        signature = self.sphincs.sign(data, self.private_key)
        sign_time_ms = (time.time() - start_time) * 1000
        signature_size = len(signature)
        return signature, sign_time_ms, signature_size

    def verify(self, data: bytes, signature: bytes) -> tuple:
        start_time = time.time()
        is_valid = self.sphincs.verify(data, signature, self.public_key)
        verify_time_ms = (time.time() - start_time) * 1000
        return is_valid, verify_time_ms
