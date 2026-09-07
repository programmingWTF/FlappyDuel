#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flappy Bird DQN 模型测试程序

功能：
- 加载通过 DQN_train.py 训练好的模型。
- 实时可视化模型玩游戏的过程。
- 在控制台输出每个回合的分数。

用法：
python DQN_test.py --model-path flappy_dqn.ckpt --episodes 10

按键：
- 按 ESC 键可随时退出。
"""

from __future__ import annotations

import argparse
import sys
import time

import pygame
import torch

# 从训练脚本中导入所需组件
try:
    from DQN_train import FlappyEnv, DuelingDQN, get_device
except ImportError:
    print("错误：无法从 'DQN_train.py' 导入。请确保 DQN_train.py 与此脚本在同一目录下。", file=sys.stderr)
    sys.exit(1)


def test(args):
    """主测试函数"""
    # 1. 设置设备和环境
    device = get_device()
    env = FlappyEnv(
        width=args.width,
        height=args.height,
        fps=args.fps,
        seed=args.seed,
        pipe_gap=args.pipe_gap,
        pipe_speed=args.pipe_speed,
        bird_radius=args.bird_radius,
    )

    # 2. 加载模型检查点
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        meta = ckpt.get("meta", {})
        print(f"成功加载模型检查点: {args.model_path}")
        print(f"  - 训练回合数: {meta.get('episodes', '未知')}")
        print(f"  - 最佳分数: {meta.get('best_score', '未知')}")
    except FileNotFoundError:
        print(f"错误: 在路径 '{args.model_path}' 未找到模型文件。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"加载模型时出错: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 创建模型并加载权重
    state_dim = meta.get("state_dim", 18)
    action_dim = meta.get("action_dim", 2)
    hidden_dim = meta.get("hidden", 256)

    policy = DuelingDQN(state_dim, action_dim, hidden=hidden_dim).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # 4. 运行评估循环
    total_scores = []
    print(f"\n开始测试，共运行 {args.episodes} 个回合...")

    for ep in range(1, args.episodes + 1):
        s = env.reset()
        done = False
        ep_score = 0
        start_time = time.time()

        while not done:
            # 贪心策略选择动作
            with torch.no_grad():
                ss = torch.from_numpy(s).unsqueeze(0).to(device)
                q_values = policy(ss)
                action = int(q_values.argmax(dim=1).item())

            # 与环境交互
            s, _, done, info = env.step(action)
            ep_score = info.get("score", 0)

            # 渲染游戏画面
            title = f"测试回合 {ep}/{args.episodes}  分数: {ep_score}"
            try:
                env.render(title_extra=title)
            except SystemExit:
                print("\n用户退出。")
                return

        total_scores.append(ep_score)
        duration = time.time() - start_time
        print(f"回合 {ep:2d} 结束 - 分数: {ep_score:3d}, 耗时: {duration:.2f}s")

    # 5. 打印最终结果
    if total_scores:
        avg_score = sum(total_scores) / len(total_scores)
        max_score = max(total_scores)
        print(f"\n测试完成。 平均分数: {avg_score:.2f}, 最高分数: {max_score}")
    
    pygame.quit()


def parse_args(argv=None):
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Flappy Bird DQN 模型测试程序")
    # 模型与测试
    p.add_argument("--model-path", type=str, default="flappy_dqn.ckpt", help="训练好的模型检查点文件路径")
    p.add_argument("--episodes", type=int, default=10, help="要运行的测试回合数")

    # 环境参数 (应与训练时保持一致)
    p.add_argument("--width", type=int, default=288)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--fps", type=int, default=60, help="游戏渲染的帧率")
    p.add_argument("--seed", type=int, default=42, help="环境的随机种子")
    p.add_argument("--pipe-gap", type=int, default=120)
    p.add_argument("--pipe-speed", type=float, default=2.5)
    p.add_argument("--bird-radius", type=int, default=12)

    return p.parse_args(argv)


def main():
    args = parse_args()
    print("测试参数:", args)
    try:
        test(args)
    except SystemExit:
        print("\n程序已退出。")
    except Exception as e:
        print(f"\n发生严重错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
