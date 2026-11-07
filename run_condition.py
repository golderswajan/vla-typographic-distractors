"""Run lerobot's standard LIBERO evaluation with the attack/attention patches installed.

Usage (all normal lerobot-eval arguments are passed straight through):
    ATTACK_CONDITION=typo ATTACK_OUT=logs/typo python run_condition.py --policy.path=... --env.type=libero ...
"""

import os

import vla_attack_patch

vla_attack_patch.install()

import lerobot.scripts.lerobot_eval as le  # noqa: E402

# lerobot renders + encodes an mp4 for the first 10 episodes of every task. Those videos show the raw
# camera (without our label) and cost time, and our own frames are already saved, so turn them off.
if os.environ.get("NO_VIDEO", "1") == "1":
    _orig_eval_policy_all = le.eval_policy_all

    def _eval_policy_all_no_video(*args, **kwargs):
        kwargs["max_episodes_rendered"] = 0
        kwargs["videos_dir"] = None
        return _orig_eval_policy_all(*args, **kwargs)

    le.eval_policy_all = _eval_policy_all_no_video

# Extension runs: shift LIBERO's fixed initial states so new episodes use unseen start scenes.
_offset = int(os.environ.get("INIT_OFFSET", "0"))
if _offset:
    import lerobot.envs.libero as _lib

    _orig_init = _lib.LiberoEnv.__init__

    def _init_with_offset(self, *args, **kwargs):
        kwargs["episode_index"] = kwargs.get("episode_index", 0) + _offset
        _orig_init(self, *args, **kwargs)

    _lib.LiberoEnv.__init__ = _init_with_offset
    print(f"[run_condition] LIBERO init-state offset = {_offset}")

if __name__ == "__main__":
    le.main()
