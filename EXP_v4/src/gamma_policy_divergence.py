"""gamma を変えると DQN の方策が MFI からどれだけ離れるかを測る診断スクリプト.

2 階層で方策を比較する:

  A. 決定レベル (本命)
     input_size=1 では状態は実質 theta_hat 一つなので, 方策は関数 theta_hat -> 出題項目 に
     還元できる. theta_hat グリッド上で各 gamma の top-1 項目 (マスク無しの純粋な決定規則) を
     出し, gamma 間および MFI との top-1 一致率と Fisher 情報量比を測る. 軌跡ドリフトを排除した
     選択規則そのものの比較.

  B. 軌跡レベル
     テスト集合で各 gamma を実走し, RMSE と, 各受検者の最終出題集合と MFI 最終集合との Jaccard を
     測る. 実際の挙動 (マスク・ドリフト込み) の比較.

使い方:
    uv run python EXP_v4/src/gamma_policy_divergence.py            # 本走
    QUICK=1 uv run python EXP_v4/src/gamma_policy_divergence.py    # 軽量スモーク
"""

import copy
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize_scalar
from scipy.stats import norm, spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get("QUICK") == "1"


def find_project_root() -> Path:
    cwd = Path.cwd()
    for root in (cwd, *cwd.parents):
        if (root / "EXP_v4" / "data" / "uncorrelated_banks").is_dir():
            return root
    raise FileNotFoundError("Could not find project root containing EXP_v4/data.")


ROOT = find_project_root()
RESULTS_DIR = ROOT / "EXP_v4" / "results" / "gamma_policy_divergence"


# --------------------------------------------------------------------------- #
# IRT / estimation primitives (現行 notebook と同一)                            #
# --------------------------------------------------------------------------- #
def RESPOND(item_para, theta, D=1):
    a, b, c = item_para[:, 0], item_para[:, 1], item_para[:, 2]
    p = (1 - c) / (1 + np.exp(-D * a * (theta - b))) + c
    return (np.random.random(size=p.shape) <= p).astype(int)


def FI(item_para, theta, D=1):
    """スカラー theta に対する各項目の Fisher 情報量, shape (n_items,)."""
    a, b, c = item_para[:, 0], item_para[:, 1], item_para[:, 2]
    return (
        D**2 * a**2 * (1 - c)
        / (c + np.exp(D * a * (theta - b)))
        / (1 + np.exp(-D * a * (theta - b))) ** 2
    )


def FI_matrix(item_para, theta_vec, D=1):
    """theta ベクトル (E,) に対する Fisher 情報量行列, shape (E, n_items)."""
    a, b, c = item_para[:, 0], item_para[:, 1], item_para[:, 2]
    th = theta_vec[:, None]
    return (
        D**2 * a**2 * (1 - c)
        / (c + np.exp(D * a * (th - b)))
        / (1 + np.exp(-D * a * (th - b))) ** 2
    )


def MLE(item_paras, resp, D=1):
    a, b, c = item_paras[:, 0], item_paras[:, 1], item_paras[:, 2]

    def neg_ll(x):
        logl = 0.0
        for i in range(len(resp)):
            p = (1 - c[i]) / (1 + np.exp(-D * a[i] * (x - b[i]))) + c[i]
            p = np.clip(p, 1e-10, 1 - 1e-10)
            logl -= resp[i] * np.log(p) + (1 - resp[i]) * np.log(1 - p)
        return logl

    result = cast(Any, minimize_scalar(neg_ll, bounds=(-4, 4), method="bounded"))
    return np.array(result.x).reshape(1)


def MLE_TEST(item_paras, resp, D=1):
    def neg_ll(x):
        logl = 0.0
        for i in range(resp_i.shape[0]):
            p = (1 - c[i]) / (1 + np.exp(-D * a[i] * (x - b[i]))) + c[i]
            p = np.clip(p, 1e-10, 1 - 1e-10)
            logl -= resp_i[i] * np.log(p) + (1 - resp_i[i]) * np.log(1 - p)
        return logl

    theta = np.zeros(resp.shape[1])
    for i in range(resp.shape[1]):
        resp_i = resp[:, i]
        a = item_paras[:, i, 0]
        b = item_paras[:, i, 1]
        c = item_paras[:, i, 2]
        result = cast(Any, minimize_scalar(neg_ll, bounds=(-4, 4), method="bounded"))
        theta[i] = result.x
    return np.expand_dims(theta, axis=0)


class Net(nn.Module):
    def __init__(self, input_size, first_hidden, second_hidden, action_space, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.out = nn.Linear(second_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.dropout(self.fc2(x))
        x = F.relu(x)
        return self.out(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


def apply_positive_constraint(model, min_value=0.0):
    for p in model.parameters():
        p.data = torch.clamp(p.data, min=min_value)


@dataclass
class Config:
    input_size: int = 1
    first_hidden: int = 50
    second_hidden: int = 30
    dropout_rate: float = 0.0
    test_length: int = 10
    gamma: float = 0.1
    memory_capacity: int = 1000
    epsilon: float = 0.1
    batch_size: int = 128
    q_network_iteration: int = 40
    learning_rate: float = 1e-3
    training_size: int = 1000
    validation_size: int = 200
    validation_interval: int = 50
    seed: int = 20260430
    validation_seed: int = 20260431
    test_seed: int = 20260432
    bank_type: str = "uncor"
    bank_id: int = 1
    prior: str = "normal"
    n_items: int = 200


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Training / evaluation (現行 notebook と同一ロジック, 明示的に net を受け渡す)     #
# --------------------------------------------------------------------------- #
def choose_action(eval_net, item_bank, action_space, item_id, state, epsilon):
    if np.random.rand() >= epsilon:
        state_t = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        item_id_t = torch.from_numpy(item_id).to(device).long()
        av = eval_net(state_t)
        av[:, item_id_t] = torch.zeros(item_id_t.shape).to(device)
        return torch.max(av, -1)[1].cpu().numpy()
    if any(item_id):
        return np.random.choice(np.delete(np.arange(action_space), item_id)).astype("int64").reshape(1)
    return np.random.choice(np.arange(action_space)).astype("int64").reshape(1)


def choose_action_test(eval_net, action_space, item_id, state):
    state_t = torch.FloatTensor(state.swapaxes(0, 1)).to(device)
    av = eval_net(state_t).detach().cpu().numpy()
    if item_id.shape[0] > 0:
        av[np.tile(np.arange(item_id.shape[1])[None, :], (item_id.shape[0], 1)), item_id] = 0.0
    return av.argmax(axis=1)


def train(cfg, eval_net, target_net, item_bank, action_space,
          training_theta, validation_theta, validation_initial_theta, validation_response_matrix):
    best_valid = None
    best_state = None
    loss_func = nn.MSELoss()
    eval_net.train()
    optimizer = optim.Adam(eval_net.parameters(), lr=cfg.learning_rate)
    memory = np.zeros((cfg.memory_capacity, cfg.input_size * 2 + 2))
    memory_counter = 0
    learn_step_counter = 0

    for j in range(cfg.training_size):
        state = np.concatenate((np.zeros(cfg.input_size - 1), np.random.rand(1) - 0.5))
        item_id = np.array([]).astype("int64")
        resp = np.array([]).astype("int64")
        for i in range(cfg.test_length):
            action = choose_action(eval_net, item_bank, action_space, item_id, state, cfg.epsilon)
            item_id = np.concatenate((item_id, action))
            resp = np.concatenate((resp, RESPOND(item_bank[action], training_theta[j])))
            reward = FI(item_bank[action, ], training_theta[j])
            if len(np.unique(resp)) == 1:
                if resp[-1] == 1:
                    next_state = np.array([state[-1] + (item_bank[:, 1].max() - state[-1]) / 2])
                else:
                    next_state = np.array([state[-1] - (state[-1] - item_bank[:, 1].min()) / 2])
            else:
                next_state = MLE(item_bank[item_id, ], resp)
            if cfg.input_size > 1:
                next_state = np.concatenate((state[-(cfg.input_size - 1):], next_state))
            memory[memory_counter % cfg.memory_capacity, :] = np.hstack((state, action, reward, next_state))
            memory_counter += 1
            state = next_state
            if memory_counter >= cfg.batch_size:
                bm = memory[np.random.choice(min(memory_counter, cfg.memory_capacity), cfg.batch_size), :]
                bs = torch.FloatTensor(bm[:, :cfg.input_size]).to(device)
                ba = torch.LongTensor(bm[:, cfg.input_size:cfg.input_size + 1].astype(int)).to(device)
                br = torch.FloatTensor(bm[:, cfg.input_size + 1:cfg.input_size + 2]).to(device)
                bns = torch.FloatTensor(bm[:, -cfg.input_size:]).to(device)
                q_eval = eval_net(bs).gather(1, ba)
                q_next = target_net(bns).detach()
                q_target = br if i == cfg.test_length - 1 else br + cfg.gamma * q_next.max(1)[0].view(cfg.batch_size, 1)
                loss = loss_func(q_eval, q_target)
                optimizer.zero_grad()
                loss.backward()
                apply_positive_constraint(eval_net)
                optimizer.step()
                learn_step_counter += 1
                if learn_step_counter % cfg.q_network_iteration == 0:
                    target_net.load_state_dict(eval_net.state_dict())

        if (j + 1) % cfg.validation_interval == 0:
            eval_net.eval()
            valid_bias = np.zeros((cfg.test_length, cfg.validation_size))
            state = np.concatenate((
                np.zeros((cfg.input_size - 1, cfg.validation_size)),
                np.expand_dims(validation_initial_theta, axis=0),
            ))
            item_id = np.array([])
            for i in range(cfg.test_length):
                action = choose_action_test(eval_net, action_space, item_id, state)
                step_resp = validation_response_matrix[np.arange(cfg.validation_size), action]
                if i == 0:
                    item_id = action[None, :]
                    resp = step_resp[None, :]
                else:
                    item_id = np.concatenate((item_id, action[None, :]))
                    resp = np.concatenate((resp, step_resp[None, :]))
                theta_0 = np.zeros(cfg.validation_size)
                idx_full = np.sum(resp, axis=0) == resp.shape[0]
                idx_zero = np.sum(resp, axis=0) == 0
                idx_norm = ~(idx_full | idx_zero)
                theta_0[idx_full] = state[-1, idx_full] + (item_bank[:, 1].max() - state[-1, idx_full]) / 2
                theta_0[idx_zero] = state[-1, idx_zero] + (item_bank[:, 1].min() - state[-1, idx_zero]) / 2
                theta_0[idx_norm] = np.squeeze(MLE_TEST(item_bank[item_id[:, idx_norm]], resp[:, idx_norm]))
                state = theta_0[None, :] if cfg.input_size == 1 else np.concatenate((state[-(cfg.input_size - 1):], theta_0[None, :]))
                valid_bias[i] = theta_0 - validation_theta
            result_valid = np.array([np.mean(valid_bias), np.sqrt(np.mean(valid_bias ** 2)), np.mean(np.abs(valid_bias))])
            if best_valid is None or result_valid[1] < best_valid[1]:
                best_valid = result_valid
                best_state = copy.deepcopy(eval_net.state_dict())
            eval_net.train()
    return best_state


def evaluate_dqn(cfg, eval_net, item_bank, action_space, theta_eval, initial_theta, response_matrix):
    """DQN 方策で CAT を実走. per-step RMSE と 最終出題集合 (E, test_length) を返す."""
    eval_net.eval()
    with torch.no_grad():
        n = len(theta_eval)
        state = np.concatenate((np.zeros((cfg.input_size - 1, n)), initial_theta[None, :]))
        item_id = np.array([])
        rmse = np.zeros(cfg.test_length)
        for i in range(cfg.test_length):
            action = choose_action_test(eval_net, action_space, item_id, state)
            step_resp = response_matrix[np.arange(n), action]
            if i == 0:
                item_id = action[None, :]
                resp = step_resp[None, :]
            else:
                item_id = np.concatenate((item_id, action[None, :]))
                resp = np.concatenate((resp, step_resp[None, :]))
            theta_0 = np.zeros([1, n])
            idx_full = np.sum(resp, axis=0) == resp.shape[0]
            idx_zero = np.sum(resp, axis=0) == 0
            idx_norm = ~(idx_full | idx_zero)
            theta_0[:, idx_full] = state[-1, idx_full] + (item_bank[:, 1].max() - state[-1, idx_full]) / 2
            theta_0[:, idx_zero] = state[-1, idx_zero] + (item_bank[:, 1].min() - state[-1, idx_zero]) / 2
            theta_0[:, idx_norm] = MLE_TEST(item_bank[item_id[:, idx_norm]], resp[:, idx_norm])
            rmse[i] = np.sqrt(np.mean((theta_0 - theta_eval) ** 2))
            state = theta_0 if cfg.input_size == 1 else np.concatenate((state[-(cfg.input_size - 1):], theta_0))
        return rmse, item_id.T.astype(int)  # (E, test_length)


def evaluate_mfi(cfg, item_bank, action_space, theta_eval, initial_theta, response_matrix):
    """MFI 方策 (現在の theta_hat で Fisher 情報量最大の未出題項目) で CAT を実走."""
    n = len(theta_eval)
    theta_hat = initial_theta.copy()
    administered = np.zeros((n, action_space), dtype=bool)
    item_id = np.zeros((cfg.test_length, n), dtype=int)
    resp = np.zeros((cfg.test_length, n), dtype=int)
    rmse = np.zeros(cfg.test_length)
    for i in range(cfg.test_length):
        fi = FI_matrix(item_bank, theta_hat)
        fi[administered] = -np.inf
        action = fi.argmax(axis=1)
        administered[np.arange(n), action] = True
        item_id[i] = action
        resp[i] = response_matrix[np.arange(n), action]
        r = resp[: i + 1]
        idx_full = r.sum(0) == (i + 1)
        idx_zero = r.sum(0) == 0
        idx_norm = ~(idx_full | idx_zero)
        theta_hat[idx_full] = theta_hat[idx_full] + (item_bank[:, 1].max() - theta_hat[idx_full]) / 2
        theta_hat[idx_zero] = theta_hat[idx_zero] + (item_bank[:, 1].min() - theta_hat[idx_zero]) / 2
        if idx_norm.any():
            theta_hat[idx_norm] = np.squeeze(
                MLE_TEST(item_bank[item_id[: i + 1][:, idx_norm]], r[:, idx_norm]), axis=0
            )
        rmse[i] = np.sqrt(np.mean((theta_hat - theta_eval) ** 2))
    return rmse, item_id.T  # (E, test_length)


# --------------------------------------------------------------------------- #
# Analysis A: 決定レベル (theta_hat グリッド上の top-1 方策)                       #
# --------------------------------------------------------------------------- #
def policy_map(eval_net, item_bank, grid):
    """各 grid theta に対する DQN の top-1 項目 (マスク無し) と Q 行列を返す."""
    with torch.no_grad():
        q = eval_net(torch.FloatTensor(grid[:, None]).to(device)).cpu().numpy()  # (G, n_items)
    return q.argmax(axis=1), q


def mfi_map(item_bank, grid):
    fi = FI_matrix(item_bank, grid)  # (G, n_items)
    return fi.argmax(axis=1), fi


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    if QUICK:
        cfg = replace(cfg, training_size=150, validation_size=80, validation_interval=50, n_items=120)
    gammas = (0.0, 0.3, 0.9) if QUICK else (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
    fixed_epsilon = 0.1

    bank_dir = ROOT / "EXP_v4" / "data" / "uncorrelated_banks"
    item_bank = np.array(pd.read_csv(bank_dir / f"item_bank_{cfg.bank_type}_{cfg.bank_id}.csv")[["a", "b", "c"]])[: cfg.n_items]
    action_space = item_bank.shape[0]

    trng = np.random.default_rng(cfg.seed)
    vrng = np.random.default_rng(cfg.validation_seed)
    training_theta = trng.standard_normal(cfg.training_size)
    validation_theta = vrng.standard_normal(cfg.validation_size)
    validation_initial_theta = vrng.random(cfg.validation_size) - 0.5
    a, b, c = item_bank[:, 0][None, :], item_bank[:, 1][None, :], item_bank[:, 2][None, :]
    vp = c + (1 - c) / (1 + np.exp(-a * (validation_theta[:, None] - b)))
    validation_response_matrix = (vrng.random(vp.shape) <= vp).astype(np.int64)

    theta_test = pd.read_csv(ROOT / "EXP_v4" / "data" / "theta_true" / f"theta_true_{cfg.bank_id}.csv")["x"].to_numpy(np.float64)
    test_rng = np.random.default_rng(cfg.test_seed)
    test_initial_theta = test_rng.random(len(theta_test)) - 0.5
    tp = c + (1 - c) / (1 + np.exp(-a * (theta_test[:, None] - b)))
    test_response_matrix = (test_rng.random(tp.shape) <= tp).astype(np.int64)

    print(f"device={device} QUICK={QUICK} bank={item_bank.shape} gammas={gammas}")

    # ---- train one model per gamma (fixed epsilon) ----
    states = {}
    for gamma in gammas:
        run_cfg = replace(cfg, gamma=gamma, epsilon=fixed_epsilon)
        set_seed(run_cfg.seed)  # 同一 seed: 差は gamma のみに帰属
        eval_net = Net(run_cfg.input_size, run_cfg.first_hidden, run_cfg.second_hidden, action_space, run_cfg.dropout_rate).to(device)
        target_net = Net(run_cfg.input_size, run_cfg.first_hidden, run_cfg.second_hidden, action_space, run_cfg.dropout_rate).to(device)
        eval_net.initialize()
        target_net.initialize()
        states[gamma] = train(run_cfg, eval_net, target_net, item_bank, action_space,
                              training_theta, validation_theta, validation_initial_theta, validation_response_matrix)
        print(f"  trained gamma={gamma}")

    grid = np.linspace(-3.0, 3.0, 241)
    w = norm.pdf(grid)
    w = w / w.sum()  # N(0,1) 重み: 母集団で意味のある theta 域を重視

    mfi_top1, mfi_fi = mfi_map(item_bank, grid)

    # ---- Analysis A: 決定レベル ----
    top1_by_gamma = {}
    rowsA = []
    net = Net(cfg.input_size, cfg.first_hidden, cfg.second_hidden, action_space, cfg.dropout_rate).to(device)
    for gamma in gammas:
        net.load_state_dict(states[gamma])
        net.eval()
        top1, q = policy_map(net, item_bank, grid)
        top1_by_gamma[gamma] = top1
        agree_mfi = float(np.sum(w * (top1 == mfi_top1)))
        # 選んだ項目の FI / MFI が選ぶ項目の FI (情報量の取りこぼし)
        fi_dqn = mfi_fi[np.arange(len(grid)), top1]
        fi_mfi = mfi_fi[np.arange(len(grid)), mfi_top1]
        fi_ratio = float(np.sum(w * (fi_dqn / fi_mfi)))
        # Q ランキングと FI ランキングの Spearman (滑らかな MFI 類似度), grid 平均
        rho = np.mean([spearmanr(q[g], mfi_fi[g]).correlation for g in range(0, len(grid), 4)])
        rowsA.append({"gamma": gamma, "top1_agree_MFI": agree_mfi, "FI_ratio_vs_MFI": fi_ratio, "Q_vs_FI_spearman": float(rho)})
    dfA = pd.DataFrame(rowsA)

    # gamma 間 top-1 一致行列
    labels = list(gammas)
    agree_mat = np.zeros((len(labels), len(labels)))
    for i, gi in enumerate(labels):
        for j, gj in enumerate(labels):
            agree_mat[i, j] = float(np.sum(w * (top1_by_gamma[gi] == top1_by_gamma[gj])))
    dfA_mat = pd.DataFrame(agree_mat, index=[f"g{g}" for g in labels], columns=[f"g{g}" for g in labels])

    # ---- Analysis B: 軌跡レベル ----
    mfi_rmse, mfi_sets = evaluate_mfi(cfg, item_bank, action_space, theta_test, test_initial_theta, test_response_matrix)
    rowsB = []
    for gamma in gammas:
        net.load_state_dict(states[gamma])
        net.eval()
        rmse, sets = evaluate_dqn(replace(cfg, gamma=gamma), net, item_bank, action_space, theta_test, test_initial_theta, test_response_matrix)
        # 最終出題集合の Jaccard (DQN gamma vs MFI), 受検者平均
        jac = []
        for e in range(sets.shape[0]):
            s1, s2 = set(sets[e].tolist()), set(mfi_sets[e].tolist())
            jac.append(len(s1 & s2) / len(s1 | s2))
        rowsB.append({
            "gamma": gamma,
            "RMSE_final": float(rmse[-1]),
            "RMSE_mean": float(rmse.mean()),
            "MFI_RMSE_final": float(mfi_rmse[-1]),
            "DQN_minus_MFI_final": float(rmse[-1] - mfi_rmse[-1]),
            "Jaccard_vs_MFI": float(np.mean(jac)),
        })
    dfB = pd.DataFrame(rowsB)

    # ---- 保存・表示 ----
    dfA.to_csv(RESULTS_DIR / "A_decision_level_vs_MFI.csv", index=False)
    dfA_mat.to_csv(RESULTS_DIR / "A_pairwise_gamma_topk_agreement.csv")
    dfB.to_csv(RESULTS_DIR / "B_trajectory_level.csv", index=False)

    pd.set_option("display.width", 120)
    print("\n=== Analysis A: 決定レベル (theta_hat グリッド, N(0,1)重み) ===")
    print("top1_agree_MFI  : DQN top-1 が MFI top-1 と一致する theta 割合")
    print("FI_ratio_vs_MFI : DQN が選ぶ項目の Fisher 情報量 / MFI が選ぶ項目 (1=同等)")
    print("Q_vs_FI_spearman: 全項目の Q 順位と FI 順位の順位相関")
    print(dfA.to_string(index=False))
    print("\n--- gamma 間 top-1 一致行列 (1=同一方策) ---")
    print(dfA_mat.to_string())
    print("\n=== Analysis B: 軌跡レベル (テスト集合) ===")
    print(dfB.to_string(index=False))
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
