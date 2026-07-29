# %% [markdown]
# # EXP026: Double DQN-Param-Raw（EAP・単一非相関バンク）
#
# EXP023の状態、項目特徴、報酬、ネットワークを維持し、TDターゲットを
# Double DQNへ変更する。
#
# - state: `[theta_hat_EAP, t/L]`
# - item features: 標準化した `[a, b, c]`
# - reward: 選択前の推定値における Fisher 情報量 `I_i(theta_hat_EAP)`
# - TD target: eval networkで次行動を選択し、target networkで評価
# - default bank: `data/uncorrelated_banks/item_bank_uncor_1.csv`

# %%
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm


def find_project_root() -> Path:
    """Find the repository root in local and Colab environments."""
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    candidates.extend(
        [
            Path("/content/Grad_Research"),
            Path("/content/drive/MyDrive/Grad_Research"),
            Path("/content/drive/MyDrive/Colab Notebooks/Grad_Research"),
        ]
    )

    for root in candidates:
        if (root / "data").is_dir() and (root / "EXP020").is_dir():
            return root

    raise FileNotFoundError(
        "Could not find the project root. Run this notebook inside the repository."
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


ROOT = find_project_root()
EXP026_DIR = ROOT / "EXP026"
MODEL_DIR = EXP026_DIR / "models"
RESULTS_DIR = EXP026_DIR / "results"
DEVICE = choose_device()

print(f"Device      : {DEVICE}")
print(f"Project root: {ROOT}")
print(f"Model dir   : {MODEL_DIR}")
print(f"Results dir : {RESULTS_DIR}")

# %% [markdown]
# ## Configuration
#
# `seed` は NumPy、Python、PyTorch に共通して適用する。
# 初期 EAP 推定値は事前分布 `N(0, 1)` の平均 `0` とする。


# %%
@dataclass
class Config:
    # Network: [theta_hat, t/L, a, b, c] -> 64 -> 64 -> 1
    state_size: int = 2
    item_feature_size: int = 3
    first_hidden: int = 64
    second_hidden: int = 64

    # Training (inherited from EXP023)
    test_length: int = 40
    gamma: float = 0.1
    memory_capacity: int = 1000
    epsilon: float = 0.1
    batch_size: int = 128
    q_network_iteration: int = 40
    learning_rate: float = 1e-3
    training_size: int = 1000
    validation_size: int = 200
    validation_interval: int = 50
    seed: int = 42

    # EAP quadrature
    n_quad: int = 61
    prior_mean: float = 0.0
    prior_std: float = 1.0

    # Bank / examinee distribution
    bank_type: str = "uncor"
    bank_id: int = 1
    prior: str = "normal"
    n_items: int = 500

    @property
    def input_size(self) -> int:
        return self.state_size + self.item_feature_size

    def validate(self) -> None:
        if self.state_size != 2:
            raise ValueError("EXP026 requires state_size=2: [theta_hat, t/L].")
        if self.item_feature_size != 3:
            raise ValueError("EXP026 Raw requires item_feature_size=3: [a, b, c].")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        if self.test_length > self.n_items:
            raise ValueError("test_length cannot exceed n_items without replacement.")
        if self.batch_size > self.memory_capacity:
            raise ValueError("batch_size cannot exceed memory_capacity.")
        if self.validation_interval > self.training_size:
            raise ValueError(
                "validation_interval must not exceed training_size; otherwise no "
                "checkpoint is selected."
            )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


cfg = Config()
cfg.validate()
set_seed(cfg.seed)
print(cfg)

# %% [markdown]
# ## 3PL response, Fisher information, and EAP estimation


# %%
def respond(
    item_para: np.ndarray, theta: np.ndarray | float, d: float = 1.0
) -> np.ndarray:
    """Generate independent Bernoulli responses for the supplied items."""
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    probability = c + (1.0 - c) / (1.0 + np.exp(-d * a * (theta - b)))
    return (np.random.random(size=probability.shape) <= probability).astype(np.int64)


def fisher_information(
    item_para: np.ndarray, theta: np.ndarray | float, d: float = 1.0
) -> np.ndarray:
    """Calculate 3PL Fisher information at theta."""
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    return (
        d**2
        * a**2
        * (1.0 - c)
        / (c + np.exp(d * a * (theta - b)))
        / (1.0 + np.exp(-d * a * (theta - b))) ** 2
    )


def eap_quadrature(
    item_paras: np.ndarray,
    responses: np.ndarray,
    n_quad: int = 61,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    d: float = 1.0,
) -> tuple[float, float]:
    """Return the EAP mean and posterior variance using grid quadrature."""
    theta_grid = np.linspace(
        prior_mean - 4.0 * prior_std,
        prior_mean + 4.0 * prior_std,
        n_quad,
    )
    log_prior = norm.logpdf(theta_grid, prior_mean, prior_std)

    a = item_paras[:, 0]
    b = item_paras[:, 1]
    c = item_paras[:, 2]
    probability = c[:, None] + (1.0 - c[:, None]) / (
        1.0 + np.exp(-d * a[:, None] * (theta_grid[None, :] - b[:, None]))
    )
    probability = np.clip(probability, 1e-10, 1.0 - 1e-10)
    log_likelihood = np.sum(
        responses[:, None] * np.log(probability)
        + (1 - responses[:, None]) * np.log(1.0 - probability),
        axis=0,
    )

    log_posterior = log_likelihood + log_prior
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()

    eap_mean = float(np.sum(theta_grid * posterior))
    eap_variance = float(np.sum((theta_grid - eap_mean) ** 2 * posterior))
    return eap_mean, eap_variance


# %% [markdown]
# ## Item-feature standardization
#
# 単一学習バンク全体から `a`, `b`, `c` の平均と母標準偏差（`ddof=0`）を
# 計算する。Fisher情報量と反応生成には元の項目パラメータを使用し、Qネットワーク
# の入力だけを標準化する。


# %%
def standardize_item_features(
    item_bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_mean = item_bank.mean(axis=0)
    item_std = item_bank.std(axis=0, ddof=0)
    if np.any(item_std <= 0.0):
        raise ValueError("All item-feature standard deviations must be positive.")
    standardized = (item_bank - item_mean) / item_std
    return standardized.astype(np.float32), item_mean, item_std


def load_experiment_data(config: Config) -> tuple[np.ndarray, np.ndarray]:
    bank_dir = {
        "uncor": ROOT / "data" / "uncorrelated_banks",
        "cor": ROOT / "data" / "correlated_banks",
    }.get(config.bank_type)
    if bank_dir is None:
        raise ValueError(f"Unsupported bank_type: {config.bank_type!r}.")

    bank_path = bank_dir / f"item_bank_{config.bank_type}_{config.bank_id}.csv"
    theta_path = ROOT / "data" / "theta_true" / f"theta_true_{config.bank_id}.csv"
    bank_frame = pd.read_csv(bank_path)
    item_bank = bank_frame.loc[:, ["a", "b", "c"]].to_numpy(dtype=np.float64)
    item_bank = item_bank[: config.n_items]
    theta_test = pd.read_csv(theta_path).loc[:, "x"].to_numpy(dtype=np.float64)

    if item_bank.shape != (config.n_items, config.item_feature_size):
        raise ValueError(
            f"Expected item bank shape {(config.n_items, config.item_feature_size)}, "
            f"got {item_bank.shape}."
        )
    return item_bank, theta_test


item_bank, theta_test = load_experiment_data(cfg)
item_features, item_feature_mean, item_feature_std = standardize_item_features(
    item_bank
)
action_space = item_bank.shape[0]
item_features_tensor = torch.from_numpy(item_features).to(DEVICE)

print(f"item bank       : {item_bank.shape}")
print(f"theta test      : {theta_test.shape}")
print(f"item mean [abc] : {item_feature_mean}")
print(f"item std  [abc] : {item_feature_std}")

# %% [markdown]
# ## Shared DQN-Param-Raw Q-network
#
# 同じMLPを全候補項目に共有し、出力形状を `[batch, n_items]` とする。


# %%
class DQNParamRaw(nn.Module):
    def __init__(
        self,
        state_size: int,
        item_feature_size: int,
        first_hidden: int,
        second_hidden: int,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.item_feature_size = item_feature_size
        self.q_network = nn.Sequential(
            nn.Linear(state_size + item_feature_size, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, 1),
        )

    def forward(
        self, states: torch.Tensor, standardized_items: torch.Tensor
    ) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != self.state_size:
            raise ValueError(
                f"states must have shape [B, {self.state_size}], got {states.shape}."
            )
        if (
            standardized_items.ndim != 2
            or standardized_items.shape[1] != self.item_feature_size
        ):
            raise ValueError(
                "standardized_items must have shape "
                f"[J, {self.item_feature_size}], got {standardized_items.shape}."
            )

        batch_size = states.shape[0]
        n_items = standardized_items.shape[0]
        expanded_states = states.unsqueeze(1).expand(-1, n_items, -1)
        expanded_items = standardized_items.unsqueeze(0).expand(batch_size, -1, -1)
        q_input = torch.cat((expanded_states, expanded_items), dim=-1)
        return self.q_network(q_input).squeeze(-1)

    def initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, n_items: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool_)
        self.next_available = np.zeros((capacity, n_items), dtype=np.bool_)
        self.counter = 0

    def __len__(self) -> int:
        return min(self.counter, self.capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        next_available: np.ndarray,
    ) -> None:
        index = self.counter % self.capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        self.next_available[index] = next_available
        self.counter += 1

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample a full batch.")
        indices = np.random.choice(len(self), batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
            self.next_available[indices],
        )


# %% [markdown]
# ## Action selection and masked Double DQN TD update


# %%
def choose_action(
    model: DQNParamRaw,
    state: np.ndarray,
    available: np.ndarray,
    standardized_items: torch.Tensor,
    epsilon: float,
) -> int:
    available_ids = np.flatnonzero(available)
    if available_ids.size == 0:
        raise ValueError("No item is available for selection.")
    if np.random.rand() < epsilon:
        return int(np.random.choice(available_ids))

    with torch.no_grad():
        state_tensor = (
            torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        )
        q_values = model(state_tensor, standardized_items).squeeze(0)
        available_tensor = torch.from_numpy(available).to(DEVICE)
        q_values = q_values.masked_fill(~available_tensor, -torch.inf)
        return int(q_values.argmax().item())


def choose_actions_greedy(
    model: DQNParamRaw,
    states: np.ndarray,
    available: np.ndarray,
    standardized_items: torch.Tensor,
) -> np.ndarray:
    if np.any(~available.any(axis=1)):
        raise ValueError("At least one examinee has no available item.")
    with torch.no_grad():
        state_tensor = torch.from_numpy(states.astype(np.float32)).to(DEVICE)
        available_tensor = torch.from_numpy(available).to(DEVICE)
        q_values = model(state_tensor, standardized_items)
        q_values = q_values.masked_fill(~available_tensor, -torch.inf)
        return q_values.argmax(dim=1).cpu().numpy()


def optimize_dqn(
    config: Config,
    eval_net: DQNParamRaw,
    target_net: DQNParamRaw,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    replay_buffer: ReplayBuffer,
    standardized_items: torch.Tensor,
) -> float:
    (
        states,
        actions,
        rewards,
        next_states,
        terminals,
        next_available,
    ) = replay_buffer.sample(config.batch_size)

    states_tensor = torch.from_numpy(states).to(DEVICE)
    actions_tensor = torch.from_numpy(actions).to(DEVICE)
    rewards_tensor = torch.from_numpy(rewards).to(DEVICE)
    next_states_tensor = torch.from_numpy(next_states).to(DEVICE)
    terminals_tensor = torch.from_numpy(terminals).to(DEVICE)
    next_available_tensor = torch.from_numpy(next_available).to(DEVICE)

    q_eval = (
        eval_net(states_tensor, standardized_items)
        .gather(1, actions_tensor.unsqueeze(1))
        .squeeze(1)
    )

    with torch.no_grad():
        # Double DQN: select with eval_net, evaluate with target_net.
        q_next_eval = eval_net(next_states_tensor, standardized_items)
        q_next_target = target_net(next_states_tensor, standardized_items)
        q_next_double = torch.zeros(config.batch_size, device=DEVICE)
        nonterminal = ~terminals_tensor
        if nonterminal.any():
            available_nonterminal = next_available_tensor[nonterminal]
            if torch.any(~available_nonterminal.any(dim=1)):
                raise RuntimeError(
                    "A nonterminal replay transition has no available item."
                )
            masked_q_next_eval = q_next_eval[nonterminal].masked_fill(
                ~available_nonterminal, -torch.inf
            )
            next_actions = masked_q_next_eval.argmax(dim=1)
            q_next_double[nonterminal] = (
                q_next_target[nonterminal]
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )

        q_target = rewards_tensor + (
            config.gamma * (~terminals_tensor).float() * q_next_double
        )

    loss = loss_function(q_eval, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


# %% [markdown]
# ## Greedy validation and test simulation


# %%
def simulate_greedy_cat(
    config: Config,
    model: DQNParamRaw,
    raw_item_bank: np.ndarray,
    standardized_items: torch.Tensor,
    true_theta: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    n_examinees = len(true_theta)
    n_items = raw_item_bank.shape[0]
    states = np.column_stack(
        (
            np.full(n_examinees, config.prior_mean, dtype=np.float32),
            np.zeros(n_examinees, dtype=np.float32),
        )
    )
    available = np.ones((n_examinees, n_items), dtype=np.bool_)
    selected_items = np.zeros((config.test_length, n_examinees), dtype=np.int64)
    responses = np.zeros((config.test_length, n_examinees), dtype=np.int64)
    theta_estimates = np.zeros((config.test_length, n_examinees), dtype=np.float64)

    model.eval()
    for step in range(config.test_length):
        actions = choose_actions_greedy(model, states, available, standardized_items)
        selected_items[step] = actions
        responses[step] = respond(raw_item_bank[actions], true_theta)
        available[np.arange(n_examinees), actions] = False

        current_theta = np.zeros(n_examinees, dtype=np.float64)
        for examinee in range(n_examinees):
            current_theta[examinee], _ = eap_quadrature(
                raw_item_bank[selected_items[: step + 1, examinee]],
                responses[: step + 1, examinee],
                n_quad=config.n_quad,
                prior_mean=config.prior_mean,
                prior_std=config.prior_std,
            )

        theta_estimates[step] = current_theta
        states = np.column_stack(
            (
                current_theta.astype(np.float32),
                np.full(
                    n_examinees,
                    (step + 1) / config.test_length,
                    dtype=np.float32,
                ),
            )
        )

    errors = theta_estimates - true_theta[np.newaxis, :]
    summary = pd.DataFrame(
        {
            "step": np.arange(1, config.test_length + 1),
            "Bias": errors.mean(axis=1),
            "RMSE": np.sqrt(np.mean(errors**2, axis=1)),
            "MAE": np.mean(np.abs(errors), axis=1),
        }
    )
    return summary, selected_items, responses, theta_estimates


def make_test_records(
    selected_items: np.ndarray,
    responses: np.ndarray,
    theta_estimates: np.ndarray,
    true_theta: np.ndarray,
) -> pd.DataFrame:
    test_length, n_examinees = selected_items.shape
    errors = theta_estimates - true_theta[np.newaxis, :]
    return pd.DataFrame(
        {
            "userID": np.repeat(np.arange(1, n_examinees + 1), test_length),
            "step": np.tile(np.arange(1, test_length + 1), n_examinees),
            "itemID": (selected_items + 1).T.reshape(-1),
            "resp": responses.T.reshape(-1),
            "theta_est": theta_estimates.T.reshape(-1),
            "bias": errors.T.reshape(-1),
        }
    )


# %% [markdown]
# ## Training

# `state=[theta_hat, t/L]` は行動選択前の状態である。選択・回答後の
# `next_state` は `[updated_theta_hat, (t+1)/L]` とする。


# %%
def train(
    config: Config,
    eval_net: DQNParamRaw,
    target_net: DQNParamRaw,
    raw_item_bank: np.ndarray,
    standardized_items: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    if config.prior == "normal":
        training_theta = np.random.randn(config.training_size)
    elif config.prior == "uniform":
        training_theta = np.random.uniform(-3.0, 3.0, config.training_size)
    else:
        raise ValueError(f"Unsupported prior: {config.prior!r}.")

    replay_buffer = ReplayBuffer(
        config.memory_capacity, config.state_size, raw_item_bank.shape[0]
    )
    optimizer = optim.Adam(eval_net.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    best_validation_rmse = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    learn_step_counter = 0
    interval_losses: list[float] = []
    train_log: list[dict[str, float | int]] = []

    eval_net.train()
    for episode, true_theta in enumerate(training_theta, start=1):
        # Before the first response, EAP equals the prior mean and t/L=0.
        state = np.array([config.prior_mean, 0.0], dtype=np.float32)
        available = np.ones(raw_item_bank.shape[0], dtype=np.bool_)
        selected_items: list[int] = []
        observed_responses: list[int] = []

        for step in range(config.test_length):
            action = choose_action(
                eval_net,
                state,
                available,
                standardized_items,
                config.epsilon,
            )

            # Reward is evaluated at the pre-response EAP estimate.
            reward = float(
                fisher_information(raw_item_bank[action : action + 1], state[0])[0]
            )
            response = int(respond(raw_item_bank[action : action + 1], true_theta)[0])
            selected_items.append(action)
            observed_responses.append(response)

            eap_mean, _ = eap_quadrature(
                raw_item_bank[np.asarray(selected_items)],
                np.asarray(observed_responses),
                n_quad=config.n_quad,
                prior_mean=config.prior_mean,
                prior_std=config.prior_std,
            )
            next_state = np.array(
                [eap_mean, (step + 1) / config.test_length], dtype=np.float32
            )
            next_available = available.copy()
            next_available[action] = False
            terminal = step + 1 == config.test_length

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                terminal,
                next_available,
            )
            state = next_state
            available = next_available

            if len(replay_buffer) >= config.batch_size:
                loss = optimize_dqn(
                    config,
                    eval_net,
                    target_net,
                    optimizer,
                    loss_function,
                    replay_buffer,
                    standardized_items,
                )
                interval_losses.append(loss)
                learn_step_counter += 1
                if learn_step_counter % config.q_network_iteration == 0:
                    target_net.load_state_dict(eval_net.state_dict())

        if episode % config.validation_interval == 0:
            validation_theta = np.random.choice(
                training_theta, size=config.validation_size
            )
            validation_summary, _, _, _ = simulate_greedy_cat(
                config,
                eval_net,
                raw_item_bank,
                standardized_items,
                validation_theta,
            )
            validation_start_step = min(7, config.test_length)
            validation_rmse = float(
                validation_summary.loc[
                    validation_summary["step"] >= validation_start_step, "RMSE"
                ].mean()
            )
            mean_train_loss = (
                float(np.mean(interval_losses)) if interval_losses else np.nan
            )
            train_log.append(
                {
                    "episode": episode,
                    "train_loss": mean_train_loss,
                    "validation_rmse_step_7_40_mean": validation_rmse,
                }
            )
            interval_losses.clear()
            print(
                f"episode {episode:4d}: train_loss={mean_train_loss:.6f}, "
                f"validation_rmse={validation_rmse:.6f}"
            )

            if validation_rmse < best_validation_rmse:
                best_validation_rmse = validation_rmse
                best_state = copy.deepcopy(eval_net.state_dict())
            eval_net.train()

    if best_state is None:
        raise RuntimeError("No checkpoint was selected during validation.")
    return best_state, pd.DataFrame(train_log)


eval_net = DQNParamRaw(
    cfg.state_size,
    cfg.item_feature_size,
    cfg.first_hidden,
    cfg.second_hidden,
).to(DEVICE)
target_net = DQNParamRaw(
    cfg.state_size,
    cfg.item_feature_size,
    cfg.first_hidden,
    cfg.second_hidden,
).to(DEVICE)
eval_net.initialize()
target_net.load_state_dict(eval_net.state_dict())

best_state, train_log = train(
    cfg, eval_net, target_net, item_bank, item_features_tensor
)
eval_net.load_state_dict(best_state)

# %% [markdown]
# ## Save model and test results
#
# チェックポイントにはモデル重みだけでなくConfigと項目標準化係数を保存する。

# %%
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

stem = (
    f"uncor_{cfg.bank_id}_Double_DQN_Param_Raw_"
    f"{cfg.prior}_gamma_{cfg.gamma}_seed_{cfg.seed}"
)
model_path = MODEL_DIR / (
    f"double_dqn_param_raw_{cfg.prior}_uncor_{cfg.bank_id}_"
    f"gamma_{cfg.gamma}_seed_{cfg.seed}.pt"
)
torch.save(
    {
        "model_state_dict": eval_net.state_dict(),
        "config": asdict(cfg),
        "item_feature_mean": item_feature_mean,
        "item_feature_std": item_feature_std,
    },
    model_path,
)
train_log.to_csv(RESULTS_DIR / f"train_log_{stem}.csv", index=False)
print(f"Model saved to: {model_path}")

test_summary, test_items, test_responses, test_theta_estimates = simulate_greedy_cat(
    cfg, eval_net, item_bank, item_features_tensor, theta_test
)
test_records = make_test_records(
    test_items, test_responses, test_theta_estimates, theta_test
)
test_records.to_csv(RESULTS_DIR / f"records_{stem}.csv", index=False)
test_summary.to_csv(RESULTS_DIR / f"summary_{stem}.csv", index=False)

print(test_summary.to_string(index=False))
print(f"Results saved to: {RESULTS_DIR}")
