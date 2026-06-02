# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models import DRQN
from utils import FI, MLE, MLE_TEST, Apply_Positive_Constraint

### If there is a GPU, set all training and testing to be done on the GPU, otherwise on the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "ShinyItemAnalysis_dataMedical"
DATASET_DIRS = {
    "ShinyItemAnalysis_dataMedical": ROOT
    / "data"
    / "real_responses_ShinyItemAnalysis_dataMedical",
    "TAM_data_ctest2": ROOT / "data" / "real_responses_TAM_data_ctest2",
}
DATA_DIR = DATASET_DIRS[DATASET_NAME]
RESULTS_DIR = ROOT / "results" / DATASET_NAME
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # DRQNwork
    embed_dim: int = 16
    lstm_hidden: int = 64
    dropout_rate: float = 0.0

    # Training
    test_length: int = 40
    gamma: float = 0.1
    memory_capacity: int = 200
    epsilon: float = 0.1
    batch_size: int = 32
    q_network_iteration: int = 40
    learning_rate: float = 1e-3
    validation_interval: int = 200

    # Evaluation
    network_id: str = "subject_200"
    
### Single-step epsilon-greedy action selection during training (carries LSTM hidden state)
def Choose_Action(prev_resp_t, hidden, item_id_arr, epsilon):
    with torch.no_grad():
        q_value, hidden = eval_net(prev_resp_t, hidden)
        # FIX: use rand() (uniform) instead of randn() (normal) for correct epsilon-greedy rate
        if np.random.rand() >= epsilon:
            qv = q_value.squeeze(0).squeeze(0).clone()
            if item_id_arr.size > 0:
                qv[torch.from_numpy(item_id_arr).to(device).long()] = -float("inf")
            action = int(qv.argmax().cpu().numpy())
        else:
            # FIX: use item_id_arr.size > 0 instead of any(item_id_arr) to correctly
            # detect non-empty arrays even when item 0 is the only selected item
            if item_id_arr.size > 0:
                action = int(
                    np.random.choice(np.delete(np.arange(action_space), item_id_arr))
                )
            else:
                action = int(np.random.choice(np.arange(action_space)))
    return action, hidden


### Batched single-step action selection for validation and testing (carries LSTM hidden state)
def Choose_Action_Test(prev_resps_t, hidden, item_id_history):
    with torch.no_grad():
        q_value, hidden = eval_net(prev_resps_t, hidden)
        q_value = q_value.squeeze(1).cpu().numpy()
        if item_id_history.shape[0] > 0:
            q_value[
                np.tile(np.arange(item_id_history.shape[1])[np.newaxis, :], (item_id_history.shape[0], 1)),
                item_id_history,
            ] = -np.inf
        action = q_value.argmax(axis=1)
    return action, hidden



### Train DRQN with episode-based experience replay
def TRAIN(cfg, training_size=None, validation_size=None):

    loss_func = nn.MSELoss()
    eval_net.train()
    optimizer = optim.Adam(eval_net.parameters(), lr=cfg.learning_rate)

    memory: list = []
    memory_idx = 0
    learn_step_counter = 0

    if training_size is None:
        training_size = train_valid_resp.shape[0]
    if validation_size is None:
        validation_size = min(200, training_size)

    for j in range(training_size):
        prev_resp_t = torch.tensor([[START_TOKEN]], dtype=torch.long, device=device)
        hidden = eval_net.init_hidden(1)

        item_id_arr = np.array([]).astype("int64")
        resp_arr = np.array([]).astype("int64")
        theta_current = np.random.rand(1) - 0.5

        ep_resps = [START_TOKEN]
        ep_actions = []
        ep_rewards = []

        for i in range(cfg.test_length):
            action, hidden = Choose_Action(prev_resp_t, hidden, item_id_arr, cfg.epsilon)

            response = int(train_valid_resp[j, action])
            reward = FI(item_bank[np.array([action]),], theta_current[-1])

            item_id_arr = np.concatenate((item_id_arr, np.array([action])))
            resp_arr = np.concatenate((resp_arr, np.array([response])))

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
                theta_current = MLE(item_bank[item_id_arr,], resp_arr)

            ep_resps.append(response)
            ep_actions.append(action)
            ep_rewards.append(float(reward[0]))

            prev_resp_t = torch.tensor([[response]], dtype=torch.long, device=device)

        episode = {
            "resps": np.asarray(ep_resps, dtype=np.int64),       # length T+1
            "actions": np.asarray(ep_actions, dtype=np.int64),   # length T
            "rewards": np.asarray(ep_rewards, dtype=np.float32), # length T
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

            q_full_eval, _ = eval_net(resps_t)
            with torch.no_grad():
                q_full_target, _ = target_net(resps_t)

            q_eval = q_full_eval[:, :-1, :].gather(2, actions_t.unsqueeze(-1)).squeeze(-1)

            q_next_all = q_full_target[:, 1:, :].clone()
            selected_mask = torch.cumsum(
                torch.nn.functional.one_hot(actions_t, num_classes=action_space), dim=1
            ).bool()
            q_next_all[selected_mask] = -float("inf")
            q_next = q_next_all.max(dim=2)[0]

            is_terminal = torch.zeros_like(rewards_t)
            is_terminal[:, -1] = 1.0
            q_target = rewards_t + cfg.gamma * q_next * (1.0 - is_terminal)

            loss = loss_func(q_eval, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # FIX: apply positive constraint after optimizer.step() so the clamp
            # acts on the updated weights, not the pre-update weights
            Apply_Positive_Constraint(eval_net)

            learn_step_counter += 1
            if learn_step_counter % cfg.q_network_iteration == 0:
                target_net.load_state_dict(eval_net.state_dict())

        ### Validation ###
        if (j + 1) % cfg.validation_interval == 0:
            eval_net.eval()
            valid_bias = np.zeros((cfg.test_length, validation_size))

            validation_id = np.random.choice(
                training_size, validation_size, replace=False
            )
            valid_theta = train_valid_theta[validation_id]
            valid_resp = train_valid_resp[validation_id, :]

            theta_state = np.random.rand(validation_size) - 0.5

            prev_resps_t = torch.full(
                (validation_size, 1), START_TOKEN, dtype=torch.long, device=device
            )
            valid_hidden = eval_net.init_hidden(validation_size)

            item_id_history = np.empty((0, validation_size), dtype=np.int64)
            resp_history = np.empty((0, validation_size), dtype=np.int64)

            for i in range(cfg.test_length):
                action, valid_hidden = Choose_Action_Test(
                    prev_resps_t, valid_hidden, item_id_history
                )
                response = valid_resp[np.arange(validation_size), action].astype(np.int64)

                item_id_history = np.concatenate((item_id_history, action[np.newaxis, :]))
                resp_history = np.concatenate((resp_history, response[np.newaxis, :]))

                theta_0 = np.zeros(validation_size)
                idx_full = np.sum(resp_history, axis=0) == resp_history.shape[0]
                idx_zero = np.sum(resp_history, axis=0) == 0
                idx_norm = np.bitwise_not(idx_full | idx_zero)
                theta_0[idx_full] = (
                    theta_state[idx_full]
                    + (item_bank[:, 1].max() - theta_state[idx_full]) / 2
                )
                theta_0[idx_zero] = (
                    theta_state[idx_zero]
                    + (item_bank[:, 1].min() - theta_state[idx_zero]) / 2
                )
                theta_0[idx_norm] = np.squeeze(
                    MLE_TEST(item_bank[item_id_history[:, idx_norm]], resp_history[:, idx_norm])
                )

                theta_state = theta_0
                valid_bias[i] = theta_0 - valid_theta

                prev_resps_t = torch.from_numpy(response).long().unsqueeze(1).to(device)

            step_valid = np.transpose(
                np.vstack(
                    (
                        np.arange(1, cfg.test_length + 1),
                        np.mean(valid_bias, axis=1),
                        np.sqrt(np.mean(valid_bias**2, axis=1)),
                        np.mean(abs(valid_bias), axis=1),
                    )
                )
            )
            print("subject: {}\n\n{}\n".format(j + 1, step_valid))

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                eval_net,
                MODEL_DIR / (
                    "drqn_" + DATASET_NAME + "_gamma_" + str(cfg.gamma)
                    + "_subject_" + str(j + 1) + ".t7"
                ),
            )
            TEST(
                cfg,
                valid_theta,
                validation_size,
                valid_resp,
                output_id="subject_" + str(j + 1),
            )

            eval_net.train()


### The function to test DRQN
def TEST(cfg, theta_test, testing_size=None, response_data=None, output_id=1):

    with torch.no_grad():
        eval_net.eval()

        if response_data is None:
            response_data = test_resp
        if testing_size is None:
            testing_size = response_data.shape[0]

        theta_state = np.random.rand(testing_size) - 0.5

        prev_resps_t = torch.full(
            (testing_size, 1), START_TOKEN, dtype=torch.long, device=device
        )
        test_hidden = eval_net.init_hidden(testing_size)

        item_id_history = np.empty((0, testing_size), dtype=np.int64)
        resp_history = np.empty((0, testing_size), dtype=np.int64)
        theta = np.empty((0, testing_size))
        dqn_step = np.zeros((1, 4))

        for i in range(cfg.test_length):
            action, test_hidden = Choose_Action_Test(
                prev_resps_t, test_hidden, item_id_history
            )
            response = response_data[np.arange(testing_size), action].astype(np.int64)

            item_id_history = np.concatenate((item_id_history, action[np.newaxis, :]))
            resp_history = np.concatenate((resp_history, response[np.newaxis, :]))

            theta_0 = np.zeros([1, testing_size])
            idx_full = np.sum(resp_history, axis=0) == resp_history.shape[0]
            idx_zero = np.sum(resp_history, axis=0) == 0
            idx_norm = np.bitwise_not(idx_full | idx_zero)
            theta_0[:, idx_full] = (
                theta_state[idx_full] + (item_bank[:, 1].max() - theta_state[idx_full]) / 2
            )
            theta_0[:, idx_zero] = (
                theta_state[idx_zero] + (item_bank[:, 1].min() - theta_state[idx_zero]) / 2
            )
            theta_0[:, idx_norm] = MLE_TEST(
                item_bank[item_id_history[:, idx_norm]], resp_history[:, idx_norm]
            )

            theta = np.concatenate((theta, theta_0))
            theta_state = theta_0[0]

            dqn_step = np.vstack(
                [
                    dqn_step,
                    np.array(
                        [
                            i + 1,
                            np.mean(theta_0 - theta_test),
                            np.sqrt(np.mean((theta_0 - theta_test) ** 2)),
                            np.mean(abs(theta_0 - theta_test)),
                        ]
                    ),
                ]
            )
            print(
                "step {:g}, bias {:.3f}, rmse {:.3f}, mae {:.3f}".format(
                    dqn_step[-1, 0], dqn_step[-1, 1], dqn_step[-1, 2], dqn_step[-1, 3]
                )
            )

            prev_resps_t = torch.from_numpy(response).long().unsqueeze(1).to(device)

        user_id = np.repeat(np.arange(1, testing_size + 1), cfg.test_length).reshape(-1, 1)
        step = np.tile(np.arange(1, cfg.test_length + 1), testing_size).reshape(-1, 1)
        item_id_out = (item_id_history + 1).transpose().reshape(-1, 1)
        resp_out = resp_history.transpose().reshape(-1, 1)
        theta_true_col = np.repeat(theta_test, cfg.test_length).reshape(-1, 1)
        theta_est = theta.transpose().reshape(-1, 1)
        bias = (theta - theta_test).transpose().reshape(-1, 1)
        dqn_data = np.hstack(
            [user_id, step, item_id_out, resp_out, theta_true_col, theta_est, bias]
        )
        dqn_data = pd.DataFrame(dqn_data).rename(
            columns={
                0: "userID",
                1: "step",
                2: "itemID",
                3: "resp",
                4: "theta_true",
                5: "theta_est",
                6: "bias",
            }
        )
        stem = f"real_bank_responses_DRQN_gamma_{cfg.gamma}_{output_id}"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dqn_data.to_csv(RESULTS_DIR / f"records_{stem}.csv", index=False)

        summary_rows = []
        for s, grp in dqn_data.groupby("step"):
            b = grp["bias"]
            r = (
                grp["theta_true"].corr(grp["theta_est"])
                if grp["theta_true"].std() > 0 and grp["theta_est"].std() > 0
                else float("nan")
            )
            summary_rows.append(
                {
                    "step": s,
                    "Bias": b.mean(),
                    "RMSE": np.sqrt((b**2).mean()),
                    "MAE": b.abs().mean(),
                    "r": r,
                }
            )
        pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"summary_{stem}.csv", index=False)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
cfg = Config()

START_TOKEN = 2  # response embedding index reserved for start-of-episode


### Load the item parameters
item_bank = np.array(pd.read_csv(DATA_DIR / "real item bank.csv")[["a", "b", "c"]])
action_space = item_bank.shape[0]

### Read the real responses and true theta for training
train_valid_resp = np.array(pd.read_csv(DATA_DIR / "real responses for training.csv"))
train_valid_theta = np.array(
    pd.read_csv(DATA_DIR / "true theta for training.csv")
).reshape(-1)

### Create two recurrent Q-networks
eval_net = DRQN(action_space, cfg.embed_dim, cfg.lstm_hidden, cfg.dropout_rate).to(device)
target_net = DRQN(action_space, cfg.embed_dim, cfg.lstm_hidden, cfg.dropout_rate).to(device)

### Initialize parameters
eval_net.initialize()
target_net.initialize()
target_net.load_state_dict(eval_net.state_dict())

### Start training (including validation)
TRAIN(cfg)


### Load the pre-trained model
eval_net = DRQN(action_space, cfg.embed_dim, cfg.lstm_hidden, cfg.dropout_rate).to(device)
eval_net = torch.load(
    MODEL_DIR / (
        "drqn_" + DATASET_NAME + "_gamma_" + str(cfg.gamma)
        + "_" + str(cfg.network_id) + ".t7"
    ),
    weights_only=False,
)

### Start testing
test_resp = np.array(pd.read_csv(DATA_DIR / "real responses for testing.csv"))
theta_test = np.array(pd.read_csv(DATA_DIR / "true theta for testing.csv")).reshape(-1)
TEST(cfg, theta_test)
