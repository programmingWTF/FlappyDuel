"""Is the AI's collision test identical to the human's?

Shared world (shared_world=True), n_envs=2: bird 0 = HUMAN, bird 1 = AI.
We deliberately shove birds INTO a pipe body and check who is judged dead.
If the AI had a laxer test, putting it in a pipe would NOT kill it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flappyrl.config import EnvConfig
from flappyrl.sim import FlappySim

BIRD_X = int(288 * 0.2)   # 57


def fresh():
    sim = FlappySim(EnvConfig(n_envs=2, shared_world=True, auto_reset=False, seed=0))
    sim.reset_all()
    return sim


def put_bird_in_pipe(sim, i):
    """Move pipe[0] over the bird and park bird i inside the TOP pipe body."""
    p = sim.pipes[0]
    p["x"] = float(BIRD_X - 10)      # pipe covers the bird's x
    sim.bird_y[i] = 30.0             # inside top rect (top_h = gap_y-60 >= 60)
    sim.bird_vel[i] = 0.0


def put_bird_in_gap(sim, i):
    """Park bird i safely in the middle of the gap."""
    p = sim.pipes[0]
    p["x"] = float(BIRD_X - 10)
    sim.bird_y[i] = float(p["gap_y"])
    sim.bird_vel[i] = 0.0


def check():
    print(f"{'scenario':<34} {'died(H,AI)':<14} {'scores':<12} verdict")

    # 1) BOTH birds in the pipe body -> both must die
    sim = fresh(); put_bird_in_pipe(sim, 0); put_bird_in_pipe(sim, 1)
    _, _, dones, info = sim.step(np.array([0, 0]))
    d = dones.tolist()
    print(f"{'both inside pipe':<34} {str(d):<14} {str(info['scores'].tolist()):<12} "
          f"{'OK' if d == [True, True] else 'FAIL'}")

    # 2) only AI (bird 1) in the pipe -> only AI must die
    sim = fresh(); put_bird_in_gap(sim, 0); put_bird_in_pipe(sim, 1)
    _, _, dones, info = sim.step(np.array([0, 0]))
    d = dones.tolist()
    print(f"{'only AI inside pipe':<34} {str(d):<14} {str(info['scores'].tolist()):<12} "
          f"{'OK' if d == [False, True] else 'FAIL'}")

    # 3) only HUMAN (bird 0) in the pipe -> only human must die
    sim = fresh(); put_bird_in_pipe(sim, 0); put_bird_in_gap(sim, 1)
    _, _, dones, info = sim.step(np.array([0, 0]))
    d = dones.tolist()
    print(f"{'only HUMAN inside pipe':<34} {str(d):<14} {str(info['scores'].tolist()):<12} "
          f"{'OK' if d == [True, False] else 'FAIL'}")

    # 4) both safe in the gap -> nobody dies
    sim = fresh(); put_bird_in_gap(sim, 0); put_bird_in_gap(sim, 1)
    _, _, dones, info = sim.step(np.array([0, 0]))
    d = dones.tolist()
    print(f"{'both safely in gap':<34} {str(d):<14} {str(info['scores'].tolist()):<12} "
          f"{'OK' if d == [False, False] else 'FAIL'}")

    # 5) identical policy -> identical fate (both same state, same world)
    sim = fresh()
    died = [None, None]
    for step in range(1, 3000):
        s = sim.states()
        # same deterministic rule for BOTH birds: flap when below gap centre
        acts = []
        for i in range(2):
            gy = sim.pipes[0]["gap_y"]
            acts.append(1 if sim.bird_y[i] > gy else 0)
        _, _, dones, _ = sim.step(np.array(acts))
        for i in range(2):
            if dones[i] and died[i] is None:
                died[i] = step
        if all(x is not None for x in died):
            break
    ok = died[0] == died[1]
    print(f"{'identical policy (H vs AI)':<34} {str(died):<14} {'':<12} "
          f"{'OK - same step' if ok else 'FAIL - asymmetric!'}")


if __name__ == "__main__":
    check()
