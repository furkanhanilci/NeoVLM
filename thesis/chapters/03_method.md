# Method

The proposed architecture is:

Camera and navigation command -> frozen Qwen3-VL -> query resampler -> structured driving rationale head -> async token cache -> staleness-aware fast policy -> IL policy -> bounded residual PPO -> VLM-guided reward -> CARLA / Leaderboard / Bench2Drive closed-loop evaluation.

