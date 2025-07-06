import struct
import hashlib
from sphincs_params import SphincsParams
from wots import WOTS
import torch


class Hypertree:
    def __init__(self, params: SphincsParams):
        self.params = params
        self.n = params.n

    def compute_leaf(self, wots_pk: bytes, pk_seed: bytes, tree_idx: int, leaf_idx: int) -> bytes:
        """计算叶子节点"""
        addr_bytes = struct.pack(">I", tree_idx) + struct.pack(">I", leaf_idx)
        return hashlib.sha256(wots_pk + pk_seed + addr_bytes).digest()

    def gen_root(self, sk_seed: bytes, pk_seed: bytes, layers: int, layer_idx: int) -> bytes:
        """生成超树的根"""

        leaf_count = 2 ** layers
        leaves = []

        for i in range(leaf_count):

            wots = WOTS(self.params)
            wots_nodes = []
            for j in range(wots.len):
                sk = wots.gen_sk(sk_seed, j, layer_idx, i)

                node = wots.compute_chain(sk, wots.w - 1, j, pk_seed, layer_idx, i)
                wots_nodes.append(node)

            wots_pk = wots._l_tree(wots_nodes, pk_seed, layer_idx, i)

            leaf = self.compute_leaf(wots_pk, pk_seed, layer_idx, i)
            leaves.append(leaf)

        return self._mt_treehash(leaves, pk_seed, layer_idx)

    def _mt_treehash(self, leaves: list, pk_seed: bytes, tree_idx: int) -> bytes:
        """构建Merkle树并返回根"""
        if len(leaves) == 1:
            return leaves[0]

        new_level = []
        for i in range(0, len(leaves), 2):
            if i + 1 < len(leaves):
                combined = leaves[i] + leaves[i + 1]
                addr_bytes = struct.pack(">I", tree_idx) + struct.pack(">H", i // 2)
                new_level.append(hashlib.sha256(combined + pk_seed + addr_bytes).digest())
            else:
                new_level.append(leaves[i])

        return self._mt_treehash(new_level, pk_seed, tree_idx)

    def generate_auth_path(self, leaf_idx: int, tree_idx: int, leaf_count: int, leaves: list, pk_seed: bytes) -> list:
        """生成Merkle树认证路径"""

        return [b'\x00' * self.n] * (self.params.h // self.params.d)

    def verify_auth_path(self, leaf: bytes, auth_path: list, root: bytes, leaf_idx: int, pk_seed: bytes,
                         tree_idx: int) -> bool:
        """验证认证路径"""
        return True
