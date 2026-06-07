# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import FI, MLE, MLE_TEST, RESPOND

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = EXP_DIR / "models"
RESULTS_DIR = EXP_DIR / "results"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
START_TOKEN = 2


@dataclass
class Config:
    embed_dim: int = 16
    lstm_hidden: int = 64
    dropout_rate: float = 0.0
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
    bank_type: str = "uncor"
    bank_id: int = 1
    prior: str = "normal"
    n_items: int = 200
    random_seed: int = 42


class LearnableInitDRQN(nn.Module):
    def __init__(self, action_space: int, embed_dim: int, lstm_hidden: int, dropout_rate: float):
        super().__init__()
        self.embed = nn.Embedding(3, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm_hidden = lstm_hidden
        self.init_h = nn.Parameter(torch.zeros(1, 1, lstm_hidden))
        self.init_c = nn.Parameter(torch.zeros(1, 1, lstm_hidden))

    def forward(self, resps: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        x = self.dropout(self.embed(resps))
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        return self.out(out), hidden

    def init_hidden(self, batch_size: int = 1):
        h = self.init_h.repeat(1, batch_size, 1).to(device)
        c = self.init_c.repeat(1, batch_size, 1).to(device)
        return (h, c)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.init_h)
        nn.init.zeros_(self.init_c)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_positive_constraint(model: LearnableInitDRQN, min_value: float = 0.0):
    out_layer = cast(nn.Linear, model.out)
    out_layer.weight.data.clamp_(min=min_value)
    if out_layer.bias is not None:
        out_layer.bias.data.clamp_(min=min_value)


def choose_action(
    model: LearnableInitDRQN,
    prev_resp_t: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
    item_id_arr: np.ndarray,
    epsilon: float,
    action_space: int,
):
    with torch.no_grad():
        q_value, hidden = model(prev_resp_t, hidden)
        qv = q_value.squeeze(0).squeeze(0).clone()
        if item_id_arr.size > 0:
            qv[torch.from_numpy(item_id_arr).to(device)] = -float("inf")
        if np.random.randn() >= epsilon:
            action = int(qv.argmax().item())
        else:
            candidates = np.setdiff1d(np.arange(action_space), item_id_arr, assume_unique=False)
            action = int(np.random.choice(candidates))
    return action, hidden


def choose_action_test(
    model: LearnableInitDRQN,
    prev_resps_t: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
    item_id_history: np.ndarray,
):
    with torch.no_grad():
        q_value, hidden = model(prev_resps_t, hidden)
        q_value = q_value.squeeze(1).cpu().numpy()
        if item_id_history.shape[0] > 0:
            row_index = np.tile(np.arange(item_id_history.shape[1])[np.newaxis, :], (item_id_history.shape[0], 1))
            q_value[row_index, item_id_history] = -np.inf
        action = q_value.argmax(axis=1)
    return action, hidden


def update_theta_state(
    item_bank: np.ndarray,
    item_id_history: np.ndarray,
    resp_history: np.ndarray,
    theta_state: np.ndarray,
):
    theta_next = np.zeros(theta_state.shape[0])
    idx_full = np.sum(resp_history, axis=0) == resp_history.shape[0]
    idx_zero = np.sum(resp_history, axis=0) == 0
    idx_norm = np.bitwise_not(idx_full | idx_zero)
    theta_next[idx_full] = theta_state[idx_full] + (item_bank[:, 1].max() - theta_state[idx_full]) / 2
    theta_next[idx_zero] = theta_state[idx_zero] + (item_bank[:, 1].min() - theta_state[idx_zero]) / 2
    if np.any(idx_norm):
        theta_next[idx_norm] = np.squeeze(
            MLE_TEST(item_bank[item_id_history[:, idx_norm]], resp_history[:, idx_norm])
        )
    return theta_next


def train(
    cfg: Config,
    item_bank: np.ndarray,
    action_space: int,
    eval_net: LearnableInitDRQN,
    target_net: LearnableInitDRQN,
):
    best_valid = None
    best_state = None

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(eval_net.parameters(), lr=cfg.learning_rate)
    eval_net.train()

    memory: list[dict[str, np.ndarray]] = []
    memory_idx = 0
    learn_step_counter = 0

    if cfg.prior == "normal":
        training_theta = np.random.randn(cfg.training_size)
    elif cfg.prior == "uniform":
        training_theta = np.random.uniform(-3, 3, cfg.training_size)
    else:
        raise ValueError(f"Unsupported prior: {cfg.prior}")

    for j in range(cfg.training_size):
        prev_resp_t = torch.tensor([[START_TOKEN]], dtype=torch.long, device=device)
        hidden = eval_net.init_hidden(1)

        item_id_arr = np.array([], dtype=np.int64)
        resp_arr = np.array([], dtype=np.int64)
        theta_current = np.random.rand(1) - 0.5

        ep_resps = [START_TOKEN]
        ep_actions = []
        ep_rewards = []

        for _ in range(cfg.test_length):
            action, hidden = choose_action(
                eval_net, prev_resp_t, hidden, item_id_arr, cfg.epsilon, action_space
            )

            response = int(RESPOND(item_bank[np.array([action])], training_theta[j])[0])
            reward = FI(item_bank[np.array([action])], training_theta[j])

            item_id_arr = np.concatenate((item_id_arr, np.array([action], dtype=np.int64)))
            resp_arr = np.concatenate((resp_arr, np.array([response], dtype=np.int64)))

            if len(np.unique(resp_arr)) == 1:
                if response == 1:
                    theta_current = np.array(
                        [theta_current[-1] + (item_bank[:, 1].max() - theta_current[-1]) / 2]
                    )
                else:
                    theta_current = np.array(
                        [theta_current[-1] - (theta_current[-1] - item_bank[:, 1].min()) / 2]
                    )
            else:
                theta_current = MLE(item_bank[item_id_arr], resp_arr)

            ep_resps.append(response)
            ep_actions.append(action)
            ep_rewards.append(float(reward[0]))
            prev_resp_t = torch.tensor([[response]], dtype=torch.long, device=device)

        episode = {
            "resps": np.asarray(ep_resps, dtype=np.int64),
            "actions": np.asarray(ep_actions, dtype=np.int64),
            "rewards": np.asarray(ep_rewards, dtype=np.float32),
        }
        if len(memory) < cfg.memory_capacity:
            memory.append(episode)
        else:
            memory[memory_idx] = episode
        memory_idx = (memory_idx + 1) % cfg.memory_capacity

        if len(memory) >= cfg.batch_size:
            indices = np.random.choice(len(memory), cfg.batch_size, replace=False)
            batch = [memory[i] for i in indices]

            resps_t = torch.LongTensor(np.stack([ep["resps"] for ep in batch])).to(device)
            actions_t = torch.LongTensor(np.stack([ep["actions"] for ep in batch])).to(device)
            rewards_t = torch.FloatTensor(np.stack([ep["rewards"] for ep in batch])).to(device)

            eval_hidden = eval_net.init_hidden(cfg.batch_size)
            q_full_eval, _ = eval_net(resps_t, eval_hidden)
            with torch.no_grad():
                target_hidden = target_net.init_hidden(cfg.batch_size)
                q_full_target, _ = target_net(resps_t, target_hidden)

            q_eval = q_full_eval[:, :-1, :].gather(2, actions_t.unsqueeze(-1)).squeeze(-1)

            q_next_all = q_full_target[:, 1:, :].clone()
            selected_mask = torch.cumsum(F.one_hot(actions_t, num_classes=action_space), dim=1).bool()
            q_next_all[selected_mask] = -float("inf")
            q_next = q_next_all.max(dim=2)[0]

            is_terminal = torch.zeros_like(rewards_t)
            is_terminal[:, -1] = 1.0
            q_target = rewards_t + cfg.gamma * q_next * (1.0 - is_terminal)

            loss = loss_func(q_eval, q_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eval_net.parameters(), max_norm=1.0)
            optimizer.step()
            apply_positive_constraint(eval_net)

            learn_step_counter += 1
            if learn_step_counter % cfg.q_network_iteration == 0:
                target_net.load_state_dict(eval_net.state_dict())

        if (j + 1) % cfg.validation_interval == 0:
            eval_net.eval()
            valid_bias = np.zeros((cfg.test_length, cfg.validation_size))
            valid_theta = np.random.choice(training_theta, cfg.validation_size)
            theta_state = np.random.rand(cfg.validation_size) - 0.5

            prev_resps_t = torch.full((cfg.validation_size, 1), START_TOKEN, dtype=torch.long, device=device)
            valid_hidden = eval_net.init_hidden(cfg.validation_size)
            item_id_history = np.empty((0, cfg.validation_size), dtype=np.int64)
            resp_history = np.empty((0, cfg.validation_size), dtype=np.int64)

            for i in range(cfg.test_length):
                action, valid_hidden = choose_action_test(eval_net, prev_resps_t, valid_hidden, item_id_history)
                response = RESPOND(item_bank[action], valid_theta).astype(np.int64)

                item_id_history = np.concatenate((item_id_history, action[np.newaxis, :]))
                resp_history = np.concatenate((resp_history, response[np.newaxis, :]))
                theta_state = update_theta_state(item_bank, item_id_history, resp_history, theta_state)
                valid_bias[i] = theta_state - valid_theta
                prev_resps_t = torch.from_numpy(response).long().unsqueeze(1).to(device)

            step_valid = np.transpose(
                np.vstack(
                    (
                        np.arange(1, cfg.test_length + 1),
                        np.mean(valid_bias, axis=1),
                        np.sqrt(np.mean(valid_bias**2, axis=1)),
                        np.mean(np.abs(valid_bias), axis=1),
                    )
                )
            )
            print(f"subject: {j + 1}\n\n{step_valid}\n")

            result_valid = np.mean(step_valid[6:, 1:], axis=0)
            if best_valid is None:
                best_valid = result_valid
                best_state = copy.deepcopy(eval_net.state_dict())
            elif (abs(result_valid[0]) < abs(best_valid[0])) and np.sum(result_valid[1:] < best_valid[1:]) == 2:
                best_valid = result_valid
                best_state = copy.deepcopy(eval_net.state_dict())

            eval_net.train()

    return best_state


def test(cfg: Config, item_bank: np.ndarray, theta_test: np.ndarray, eval_net: LearnableInitDRQN):
    with torch.no_grad():
        eval_net.eval()
        testing_size = len(theta_test)
        theta_state = np.random.rand(testing_size) - 0.5

        prev_resps_t = torch.full((testing_size, 1), START_TOKEN, dtype=torch.long, device=device)
        test_hidden = eval_net.init_hidden(testing_size)

        item_id_history = np.empty((0, testing_size), dtype=np.int64)
        resp_history = np.empty((0, testing_size), dtype=np.int64)
        step_rows = []
        theta_all = []

        for i in range(cfg.test_length):
            action, test_hidden = choose_action_test(eval_net, prev_resps_t, test_hidden, item_id_history)
            response = RESPOND(item_bank[action], theta_test).astype(np.int64)

            item_id_history = np.concatenate((item_id_history, action[np.newaxis, :]))
            resp_history = np.concatenate((resp_history, response[np.newaxis, :]))
            theta_state = update_theta_state(item_bank, item_id_history, resp_history, theta_state)
            theta_all.append(theta_state.copy())

            bias = theta_state - theta_test
            row = [i + 1, np.mean(bias), np.sqrt(np.mean(bias**2)), np.mean(np.abs(bias))]
            step_rows.append(row)
            print("step {:g}, bias {:.3f}, rmse {:.3f}, mae {:.3f}".format(*row))

            prev_resps_t = torch.from_numpy(response).long().unsqueeze(1).to(device)

    theta_matrix = np.stack(theta_all, axis=0)
    user_id_col = np.repeat(np.arange(1, testing_size + 1), cfg.test_length).reshape(-1, 1)
    step_col = np.tile(np.arange(1, cfg.test_length + 1), testing_size).reshape(-1, 1)
    item_id_col = (item_id_history + 1).transpose().reshape(-1, 1)
    resp_col = resp_history.transpose().reshape(-1, 1)
    theta_est_col = theta_matrix.transpose().reshape(-1, 1)
    bias_col = (theta_matrix - theta_test).transpose().reshape(-1, 1)

    records = pd.DataFrame(
        np.hstack([user_id_col, step_col, item_id_col, resp_col, theta_est_col, bias_col]),
        columns=["userID", "step", "itemID", "resp", "theta_est", "bias"],
    )
    summary = pd.DataFrame(step_rows, columns=["step", "Bias", "RMSE", "MAE"])

    stem = f"{cfg.bank_type}_{cfg.bank_id}_DRQN_learnable_init_{cfg.prior}_gamma_{cfg.gamma}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records.to_csv(RESULTS_DIR / f"records_{stem}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"summary_{stem}.csv", index=False)


def main():
    cfg = Config()
    set_seed(cfg.random_seed)

    bank_dir = {
        "uncor": ROOT / "data" / "uncorrelated_banks",
        "cor": ROOT / "data" / "correlated_banks",
    }[cfg.bank_type]
    item_bank = np.array(
        pd.read_csv(bank_dir / f"item_bank_{cfg.bank_type}_{cfg.bank_id}.csv")[["a", "b", "c"]]
    )[: cfg.n_items]
    action_space = item_bank.shape[0]
    theta_test = np.array(pd.read_csv(ROOT / "data" / "theta_true" / f"theta_true_{cfg.bank_id}.csv")["x"])

    print(f"Config:\n{asdict(cfg)}")
    eval_net = LearnableInitDRQN(action_space, cfg.embed_dim, cfg.lstm_hidden, cfg.dropout_rate).to(device)
    target_net = LearnableInitDRQN(action_space, cfg.embed_dim, cfg.lstm_hidden, cfg.dropout_rate).to(device)
    eval_net.initialize()
    target_net.initialize()
    target_net.load_state_dict(eval_net.state_dict())

    best_state = train(cfg, item_bank, action_space, eval_net, target_net)
    if best_state is None:
        raise RuntimeError("No checkpoint was selected during validation.")

    eval_net.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"drqn_learnable_init_{cfg.prior}_{cfg.bank_type}_{cfg.bank_id}_gamma_{cfg.gamma}.pt"
    torch.save(eval_net.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    test(cfg, item_bank, theta_test, eval_net)


if __name__ == "__main__":
    main()
