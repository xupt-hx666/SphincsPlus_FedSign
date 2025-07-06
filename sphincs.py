from sphincs_params import SphincsParams
from fors import FORS
from wots import WOTS
from hypertree import Hypertree
import os
import struct
import hashlib
import torch



class SPHINCSPlus:
    def __init__(self, security_level=128):
        self.params = SphincsParams(security_level)
        self.n = self.params.n
        self.fors = FORS(self.params)
        self.wots = WOTS(self.params)
        self.ht = Hypertree(self.params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def keygen(self) -> tuple[bytes, bytes]:
        """生成密钥对"""
        sk_seed = os.urandom(self.n)
        pk_seed = os.urandom(self.n)

        root = self.ht.gen_root(sk_seed, pk_seed, self.params.d, 0)

        public_key = pk_seed + root
        private_key = sk_seed + public_key
        return public_key, private_key

    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """生成签名"""

        if len(private_key) < 3 * self.n:
            raise ValueError("无效私钥长度")

        sk_seed = private_key[:self.n]
        pk_seed = private_key[self.n:2 * self.n]
        root = private_key[2 * self.n:3 * self.n]

        rand = os.urandom(self.n)
        msg_hash = hashlib.sha256(rand + root + message).digest()

        idx = struct.unpack(">I", msg_hash[:4])[0]
        tree_idx = idx % (2 ** (self.params.h // self.params.d))
        leaf_idx = (idx >> (self.params.h // self.params.d)) % (2 ** (self.params.h // self.params.d))

        fors_start = self.params.k * self.params.a // 8
        fors_md = msg_hash[self.n // 2: self.n // 2 + fors_start]
        fors_sig = self.fors.sign(fors_md, sk_seed, pk_seed, tree_idx, leaf_idx)

        fors_pk = self.fors.pk_from_sig(fors_sig, fors_md, pk_seed, tree_idx, leaf_idx)

        wots_sig = self.wots.sign(fors_pk, sk_seed, pk_seed, tree_idx, leaf_idx)

        return (rand +
                struct.pack(">I", tree_idx) +
                struct.pack(">I", leaf_idx) +
                fors_sig +
                wots_sig)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """验证签名"""
        if len(public_key) != 2 * self.n:
            return False

        if len(signature) < 3 * self.n:
            return False

        pk_seed = public_key[:self.n]
        root = public_key[self.n:]

        rand = signature[:self.n]
        tree_idx = struct.unpack(">I", signature[self.n:self.n + 4])[0]
        leaf_idx = struct.unpack(">I", signature[self.n + 4:self.n + 8])[0]

        fors_sig_len = self.params.k * self.n
        fors_sig = signature[self.n + 8:self.n + 8 + fors_sig_len]
        wots_sig = signature[self.n + 8 + fors_sig_len:]

        msg_hash = hashlib.sha256(rand + root + message).digest()

        fors_start = self.params.k * self.params.a // 8
        fors_md = msg_hash[self.n // 2: self.n // 2 + fors_start]
        fors_pk = self.fors.pk_from_sig(fors_sig, fors_md, pk_seed, tree_idx, leaf_idx)

        wots_pk = self.wots.pk_from_sig(wots_sig, fors_pk, pk_seed, tree_idx, leaf_idx)

        leaf = self.ht.compute_leaf(wots_pk, pk_seed, tree_idx, leaf_idx)

        auth_path = self.ht.generate_auth_path(leaf_idx, tree_idx, 2 ** (self.params.h // self.params.d), [leaf],
                                               pk_seed)
        return self.ht.verify_auth_path(leaf, auth_path, root, leaf_idx, pk_seed, tree_idx)
