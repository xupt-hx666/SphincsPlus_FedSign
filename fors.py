import struct
import hashlib
from sphincs_params import SphincsParams


class FORS:
    def __init__(self, params: SphincsParams):
        self.params = params
        self.n = params.n
        self.k = params.k
        self.t = params.t
        self.a = params.a

    def sign(self, md: bytes, sk_seed: bytes, pk_seed: bytes, tree_idx: int, leaf_idx: int) -> bytes:
        """生成FORS签名"""
        sig = b""
        for i in range(self.k):
            offset = (i * self.a) // 8
            bits_offset = (i * self.a) % 8
            idx_bits = (md[offset] >> bits_offset) & (2 ** min(8 - bits_offset, self.a) - 1)

            sk = self.gen_sk(sk_seed, i, tree_idx, leaf_idx)

            sig += sk
        return sig

    def gen_sk(self, sk_seed: bytes, idx: int, tree_idx: int, leaf_idx: int) -> bytes:
        """生成FORS私钥"""
        addr_bytes = struct.pack(">I", tree_idx) + struct.pack(">I", leaf_idx) + struct.pack(">H", idx)
        return hashlib.sha256(sk_seed + b'\x00' + addr_bytes).digest()

    def pk_from_sig(self, sig: bytes, md: bytes, pk_seed: bytes, tree_idx: int, leaf_idx: int) -> bytes:
        """从签名恢复FORS公钥"""
        roots = []
        for i in range(self.k):
            offset = (i * self.a) // 8
            bits_offset = (i * self.a) % 8
            leaf_idx = (md[offset] >> bits_offset) & (2 ** min(8 - bits_offset, self.a) - 1)

            sk = sig[i * self.n: (i + 1) * self.n]

            node = self.compute_leaf_node(sk, i, tree_idx, leaf_idx, pk_seed)
            roots.append(node)

        return hashlib.sha256(b''.join(roots)).digest()

    def compute_leaf_node(self, sk: bytes, idx: int, tree_idx: int, leaf_idx: int, pk_seed: bytes) -> bytes:
        """计算叶子节点"""
        return hashlib.sha256(
            sk + pk_seed + struct.pack(">I", tree_idx) + struct.pack(">I", leaf_idx) + struct.pack(">H", idx)).digest()
