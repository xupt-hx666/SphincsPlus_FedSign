from model import MedModel
from args import args_parser
from client import train, validate
import torch.nn as nn
from crypto import SphincsCPU
import torch
import numpy as np
import time

args = args_parser()


class FedPer:
    def __init__(self):
        self.args = args

        if args.use_sphincs:
            self.signer = SphincsCPU(security_level=args.sphincs_security)
        else:
            self.signer = None

        self.global_base = self.base_layers = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(args.device)

        self.client_models = []
        for _ in range(args.K):
            model = MedModel(name=f"client_{_}").to(args.device)
            model.base_layers.load_state_dict(self.global_base.state_dict())
            self.client_models.append(model)

        self.sign_stats = {
            'times_ms': [],
            'sizes': [],
            'verify_times_ms': []
        }
        self.round_stats = []

    def aggregate(self, client_models):
        """联邦聚合"""
        global_dict = self.global_base.state_dict()

        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])

        for model in client_models:
            model_dict = model.base_layers.state_dict()
            for key in global_dict:
                if key in model_dict:
                    global_dict[key] += model_dict[key].to(args.device)

        for key in global_dict:
            global_dict[key] = global_dict[key] / len(client_models)

        return global_dict

    def server_round(self, round_idx):
        num_selected = max(int(args.C * args.K), 1)
        selected_clients = np.random.choice(range(args.K), num_selected, replace=False)
        print(f"Round {round_idx + 1}: Selected clients: {selected_clients}")

        trained_models = []
        round_sign_times_ms = []
        round_sign_sizes = []
        round_verify_times_ms = []

        for client_id in selected_clients:
            model = self.client_models[client_id]
            model.base_layers.load_state_dict(self.global_base.state_dict())

            trained_model = train(args, model, client_id)
            trained_models.append(trained_model)

            if self.signer:
                weight_data = b''
                weights = trained_model.base_layers.state_dict()
                for key, value in weights.items():
                    weight_data += value.cpu().numpy().tobytes()

                signature, sign_time_ms, sign_size = self.signer.sign(weight_data)
                round_sign_times_ms.append(sign_time_ms)
                round_sign_sizes.append(sign_size)

                is_valid, verify_time_ms = self.signer.verify(weight_data, signature)
                round_verify_times_ms.append(verify_time_ms)

                print(f"Client {client_id} | "
                      f"签名时间: {sign_time_ms:.2f}ms | "
                      f"验证时间: {verify_time_ms:.2f}ms | "
                      f"签名大小: {sign_size} bytes | "
                      f"验证结果: {'成功' if is_valid else '失败'}")

        self.sign_stats['times_ms'].extend(round_sign_times_ms)
        self.sign_stats['sizes'].extend(round_sign_sizes)
        self.sign_stats['verify_times_ms'].extend(round_verify_times_ms)

        round_stat = {
            'round': round_idx + 1,
            'avg_sign_time_ms': np.mean(round_sign_times_ms) if round_sign_times_ms else 0,
            'avg_verify_time_ms': np.mean(round_verify_times_ms) if round_verify_times_ms else 0,
            'avg_sign_size': np.mean(round_sign_sizes) if round_sign_sizes else 0
        }
        self.round_stats.append(round_stat)

        print(f"\nRound {round_idx + 1} SPHINCS+ 统计:")
        print(f"平均签名时间: {round_stat['avg_sign_time_ms']:.2f}ms")
        print(f"平均验证时间: {round_stat['avg_verify_time_ms']:.2f}ms")
        print(f"平均签名大小: {round_stat['avg_sign_size']:.2f} bytes")

        global_weights = self.aggregate(trained_models)
        self.global_base.load_state_dict(global_weights)

        for model in self.client_models:
            model.base_layers.load_state_dict(self.global_base.state_dict())

        val_accs = []
        for client_id in selected_clients:
            acc = validate(args, self.client_models[client_id], client_id)
            val_accs.append(acc)
            print(f"Client {client_id} Val Acc: {acc:.2f}%")

        avg_acc = sum(val_accs) / len(val_accs)
        print(f"Round {round_idx + 1} Average Val Acc: {avg_acc:.2f}%")
        return avg_acc

    def _print_final_stats(self):
        if not self.sign_stats['times_ms']:
            return

        print("\n" + "=" * 50)
        print("SPHINCS+ 全局统计:")
        print(f"总签名次数: {len(self.sign_stats['times_ms'])}")
        print(f"平均签名时间: {np.mean(self.sign_stats['times_ms']):.2f}ms")
        print(f"平均验证时间: {np.mean(self.sign_stats['verify_times_ms']):.2f}ms")
        print(f"平均签名大小: {np.mean(self.sign_stats['sizes']):.2f} bytes")
        print(f"最大签名时间: {np.max(self.sign_stats['times_ms']):.2f}ms")
        print(f"最小签名时间: {np.min(self.sign_stats['times_ms']):.2f}ms")
        print(f"最大签名大小: {np.max(self.sign_stats['sizes'])} bytes")
        print(f"最小签名大小: {np.min(self.sign_stats['sizes'])} bytes")
        print("=" * 50 + "\n")

        print("每轮详细统计:")
        print("轮次 | 平均签名时间(ms) | 平均验证时间(ms) | 平均签名大小(bytes)")
        for stat in self.round_stats:
            print(f"{stat['round']:2d} | "
                  f"{stat['avg_sign_time_ms']:.2f} | "
                  f"{stat['avg_verify_time_ms']:.2f} | "
                  f"{stat['avg_sign_size']:.2f}")

    def run(self):
        if self.signer:
            print(f"\nSPHINCS+初始化完成 | 安全级别: {args.sphincs_security} | "
                  f"密钥生成时间: {self.signer.keygen_time_ms:.2f}ms")

        for r in range(args.r):
            print(f"\n=== Round {r + 1}/{args.r} ===")
            self.server_round(r)
        self._print_final_stats()

