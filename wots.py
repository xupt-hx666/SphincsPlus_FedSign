from sphincs_params import SphincsParams
import struct
import hashlib


class WOTS:
    def __init__(self, params: SphincsParams):
        self.params = params
        self.n = params.n
        self.w = params.w
        self.len = params.len
        self.len1 = params.len1
        self.len2 = params.len2

    def sign(self, msg: bytes, sk_seed: bytes, pk_seed: bytes, tree_idx: int, leaf_idx: int) -> bytes:
        """生成WOTS+签名"""
        msg_vals = self._chain_lengths(msg)
        csum = 0
        for val in msg_vals[:self.len1]:
            csum += self.w - 1 - val

        csum_vals = []
        for _ in range(self.len2):
            csum_vals.append(csum % self.w)
            csum //= self.w

        sig = b""
        for i in range(self.len):
            if i < self.len1:
                chain_len = msg_vals[i]
            else:
                chain_len = csum_vals[i - self.len1]

            sk = self.gen_sk(sk_seed, i, tree_idx, leaf_idx)

            node = self.compute_chain(sk, chain_len, i, pk_seed, tree_idx, leaf_idx)
            sig += node

        return sig

    def gen_sk(self, sk_seed: bytes, idx: int, tree_idx: int, leaf_idx: int) -> bytes:
        """生成私钥元素"""
        addr_bytes = struct.pack(">I", tree_idx) + struct.pack(">I", leaf_idx) + struct.pack(">H", idx)
        return hashlib.sha256(sk_seed + b'\x00' + addr_bytes).digest()

    def compute_chain(self, start: bytes, steps: int, key_idx: int, pk_seed: bytes, tree_idx: int,
                      leaf_idx: int) -> bytes:
        """计算哈希链"""
        node = start
        for i in range(steps):
            node = self._hash(node, i, key_idx, pk_seed, tree_idx, leaf_idx)
        return node

    def _hash(self, node: bytes, chain_index: int, key_index: int, pk_seed: bytes, tree_idx: int,
              leaf_idx: int) -> bytes:
        """链上的哈希函数"""
        addr_bytes = struct.pack(">I", tree_idx) + struct.pack(">I", leaf_idx) + struct.pack(">H",
                                                                                             key_index) + struct.pack(
            ">H", chain_index)
        return hashlib.sha256(node + pk_seed + addr_bytes).digest()

    def _chain_lengths(self, data: bytes) -> list:
        """计算链长表示"""
        vals = []
        for byte in data:
            vals.append(byte % self.w)
            vals.append(byte // self.w)

        while len(vals) < self.len1:
            vals.append(0)

        return vals[:self.len1]

    def pk_from_sig(self, sig: bytes, msg: bytes, pk_seed: bytes, tree_idx: int, leaf_idx: int) -> bytes:
        """从签名恢复WOTS+公钥"""

        nodes = [sig[i * self.n: (i + 1) * self.n] for i in range(self.len)]

        msg_vals = self._chain_lengths(msg)
        csum = 0
        for val in msg_vals[:self.len1]:
            csum += self.w - 1 - val

        csum_vals = []
        for _ in range(self.len2):
            csum_vals.append(csum % self.w)
            csum //= self.w

        wots_pk = []
        for i in range(self.len):
            if i < self.len1:
                steps_remaining = self.w - 1 - msg_vals[i]
            else:
                steps_remaining = self.w - 1 - csum_vals[i - self.len1]

            node = nodes[i]
            if steps_remaining > 0:
                node = self.compute_chain(node, steps_remaining, i, pk_seed, tree_idx, leaf_idx)
            wots_pk.append(node)

        return self._l_tree(wots_pk, pk_seed, tree_idx, leaf_idx)

    def _l_tree(self, nodes: list, pk_seed: bytes, tree_idx: int, leaf_idx: int) -> bytes:
        """使用L-tree压缩公钥"""
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    combined = nodes[i] + nodes[i + 1]
                    addr_bytes = struct.pack(">I", tree_idx) + struct.pack(">I", leaf_idx) + struct.pack(">H", i)
                    new_nodes.append(hashlib.sha256(combined + pk_seed + addr_bytes).digest())
                else:
                    new_nodes.append(nodes[i])
            nodes = new_nodes

        return nodes[0]
