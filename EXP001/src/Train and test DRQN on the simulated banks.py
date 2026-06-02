# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import DRQN

ROOT = Path(__file__).resolve().parents[2]
EXP001_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = EXP001_DIR / "models"
RESULTS_DIR = EXP001_DIR / "results"

### If there is a GPU, set all training and testing to be done on the GPU, otherwise on the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

START_TOKEN = 2  # response embedding index reserved for start-of-episode


### The function to simulate the examinee and generate the response under 3PLM
def RESPOND(item_para, theta, D=1):
    a = item_para[:,0]
    b = item_para[:,1]
    c = item_para[:,2]
    p = (1-c) / (1 + np.exp(-D*a*(theta-b))) + c
    resp = (np.random.rand(1) <= p).astype(int)
    return resp

### The function to calculate the Fisher item information under 3PLM
def FI(item_para, theta, D=1):
    a = item_para[:,0]
    b = item_para[:,1]
    c = item_para[:,2]
    info = D**2*a**2*(1-c) / (c+np.exp(D*a*(theta-b))) / (1+np.exp(-D*a*(theta-b)))**2
    return info

### The function to deploy maximum likelihood estimation under 3PLM for training phase
def MLE(item_paras, resp, D=1):
    a = item_paras[:,0]
    b = item_paras[:,1]
    c = item_paras[:,2]
    def mins_likelihood(x):
        logl = 0
        for i in range(len(resp)):
            p = (1-c[i]) / (1 + np.exp(-D*a[i]*(x - b[i]))) + c[i]
            logl -= resp[i] * np.log(p) + (1-resp[i]) * np.log((1-p))
        return logl
    theta = minimize_scalar(mins_likelihood, bounds=(-4, 4), method='Bounded').x
    return np.array(theta).reshape(1,)

### The function to parallelly deploy maximum likelihood estimation under 3PLM for validation and testing phases
def MLE_TEST(item_paras, resp, D=1):
    def mins_likelihood(x):
        logl = 0
        for i in range(resp_i.shape[0]):
            p = (1-c[i]) / (1 + np.exp(-D*a[i]*(x - b[i]))) + c[i]
            logl -= resp_i[i] * np.log(p) + (1-resp_i[i]) * np.log((1-p))
        return logl
    theta = np.zeros(resp.shape[1])
    for i in range(resp.shape[1]):
        resp_i = resp[:,i]
        a = item_paras[:,i,0]
        b = item_paras[:,i,1]
        c = item_paras[:,i,2]
        theta[i] = minimize_scalar(mins_likelihood, bounds=(-4, 4), method='Bounded').x
    return np.expand_dims(theta, axis=0)

### The function that limits the network parameters to positive, such that the output of Q-network (the expectation of cumulative information) is positive
def Apply_Positive_Constraint(model, min_value=0.0):
    for param in model.parameters():
        param.data = torch.clamp(param.data, min=min_value)


### Single-step epsilon-greedy action selection during training (carries LSTM hidden state)
def Choose_Action(prev_resp_t, hidden, item_id_arr, epsilon):
    with torch.no_grad():
        q_value, hidden = eval_net(prev_resp_t, hidden)
        if np.random.randn() >= epsilon:
            qv = q_value.squeeze(0).squeeze(0).clone()
            if any(item_id_arr):
                qv[torch.from_numpy(item_id_arr).to(device).long()] = torch.zeros(item_id_arr.shape).to(device)
            action = int(qv.argmax().cpu().numpy())
        else:
            if any(item_id_arr):
                action = int(np.random.choice(np.delete(np.arange(action_space), item_id_arr)))
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
            ] = np.zeros(item_id_history.shape)
        action = q_value.argmax(axis=1)
    return action, hidden


### The function to train DRQN including validation
def TRAIN(gamma,
          prior,
          memory_capacity=200,
          epsilon=0.1,
          batch_size=32,
          q_network_iteration=40,
          learning_rate=1e-3,
          training_size=1000,
          validation_size=200,
          validation_interval=50):

    global best_valid

    loss_func = nn.MSELoss()
    eval_net.train()
    optimizer = optim.Adam(eval_net.parameters(), lr=learning_rate)

    memory: list = []
    memory_idx = 0
    learn_step_counter = 0

    if prior == "normal":
        training_theta = np.random.randn(training_size)
    elif prior == "uniform":
        training_theta = np.random.uniform(-3, 3, training_size)

    for j in range(training_size):
        prev_resp_t = torch.tensor([[START_TOKEN]], dtype=torch.long, device=device)
        hidden = eval_net.init_hidden(1)

        item_id_arr = np.array([]).astype("int64")
        resp_arr = np.array([]).astype("int64")
        theta_current = np.random.rand(1) - 0.5

        ep_resps = [START_TOKEN]
        ep_actions = []
        ep_rewards = []

        for i in range(test_length):
            action, hidden = Choose_Action(prev_resp_t, hidden, item_id_arr, epsilon)

            response = int(RESPOND(item_bank[np.array([action])], training_theta[j])[0])
            reward = FI(item_bank[np.array([action])], theta_current[-1])

            item_id_arr = np.concatenate((item_id_arr, np.array([action])))
            resp_arr = np.concatenate((resp_arr, np.array([response])))

            if len(np.unique(resp_arr)) == 1:
                if response == 1:
                    theta_current = np.array([theta_current[-1] + (item_bank[:,1].max() - theta_current[-1]) / 2])
                else:
                    theta_current = np.array([theta_current[-1] - (theta_current[-1] - item_bank[:,1].min()) / 2])
            else:
                theta_current = MLE(item_bank[item_id_arr], resp_arr)

            ep_resps.append(response)
            ep_actions.append(action)
            ep_rewards.append(float(reward[0]))

            prev_resp_t = torch.tensor([[response]], dtype=torch.long, device=device)

        episode = {
            "resps":   np.asarray(ep_resps,   dtype=np.int64),    # length T+1
            "actions": np.asarray(ep_actions, dtype=np.int64),    # length T
            "rewards": np.asarray(ep_rewards, dtype=np.float32),  # length T
        }
        if len(memory) < memory_capacity:
            memory.append(episode)
        else:
            memory[memory_idx] = episode
        memory_idx = (memory_idx + 1) % memory_capacity

        if len(memory) >= batch_size:
            indices = np.random.choice(len(memory), batch_size, replace=False)
            batch = [memory[i] for i in indices]

            resps_t   = torch.LongTensor(np.stack([ep["resps"]   for ep in batch])).to(device)
            actions_t = torch.LongTensor(np.stack([ep["actions"] for ep in batch])).to(device)
            rewards_t = torch.FloatTensor(np.stack([ep["rewards"] for ep in batch])).to(device)

            q_full_eval, _ = eval_net(resps_t)
            with torch.no_grad():
                q_full_target, _ = target_net(resps_t)

            q_eval = q_full_eval[:, :-1, :].gather(2, actions_t.unsqueeze(-1)).squeeze(-1)

            q_next_all = q_full_target[:, 1:, :].clone()
            selected_mask = torch.cumsum(
                F.one_hot(actions_t, num_classes=action_space), dim=1
            ).bool()
            q_next_all[selected_mask] = -float("inf")
            q_next = q_next_all.max(dim=2)[0]

            is_terminal = torch.zeros_like(rewards_t)
            is_terminal[:, -1] = 1.0
            q_target = rewards_t + gamma * q_next * (1.0 - is_terminal)

            loss = loss_func(q_eval, q_target)
            optimizer.zero_grad()
            loss.backward()
            Apply_Positive_Constraint(eval_net)
            optimizer.step()

            learn_step_counter += 1
            if learn_step_counter % q_network_iteration == 0:
                target_net.load_state_dict(eval_net.state_dict())


        ### Validation ###
        if (j+1) % validation_interval == 0:
            eval_net.eval()
            valid_bias = np.zeros((test_length, validation_size))
            valid_theta = np.random.choice(training_theta, validation_size)

            theta_state = np.random.rand(validation_size) - 0.5

            prev_resps_t = torch.full(
                (validation_size, 1), START_TOKEN, dtype=torch.long, device=device
            )
            valid_hidden = eval_net.init_hidden(validation_size)

            item_id_history = np.empty((0, validation_size), dtype=np.int64)
            resp_history    = np.empty((0, validation_size), dtype=np.int64)

            for i in range(test_length):
                action, valid_hidden = Choose_Action_Test(
                    prev_resps_t, valid_hidden, item_id_history
                )
                response = RESPOND(item_bank[action], valid_theta).astype(np.int64)

                item_id_history = np.concatenate((item_id_history, action[np.newaxis, :]))
                resp_history    = np.concatenate((resp_history,    response[np.newaxis, :]))

                theta_0 = np.zeros(validation_size)
                idx_full = np.sum(resp_history, axis=0) == resp_history.shape[0]
                idx_zero = np.sum(resp_history, axis=0) == 0
                idx_norm = np.bitwise_not(idx_full | idx_zero)
                theta_0[idx_full] = theta_state[idx_full] + (item_bank[:,1].max() - theta_state[idx_full]) / 2
                theta_0[idx_zero] = theta_state[idx_zero] + (item_bank[:,1].min() - theta_state[idx_zero]) / 2
                theta_0[idx_norm] = np.squeeze(
                    MLE_TEST(item_bank[item_id_history[:,idx_norm]], resp_history[:,idx_norm])
                )

                theta_state = theta_0
                valid_bias[i] = theta_0 - valid_theta

                prev_resps_t = torch.from_numpy(response).long().unsqueeze(1).to(device)

            step_valid = np.transpose(np.vstack((
                np.arange(1, test_length+1),
                np.mean(valid_bias, axis=1),
                np.sqrt(np.mean(valid_bias**2, axis=1)),
                np.mean(abs(valid_bias), axis=1),
            )))

            print("subject: {}\n\n{}\n".format(j+1, step_valid))

            result_valid = np.mean(step_valid[6:, 1:], axis=0)

            model_path = MODEL_DIR / f"drqn_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.t7"
            try:
                best_valid
            except NameError:
                best_valid = result_valid
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(eval_net, model_path)

            if (abs(result_valid[0]) < abs(best_valid[0])) & np.sum(result_valid[1:] < best_valid[1:]) == 2:
                best_valid = result_valid
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(eval_net, model_path)

            eval_net.train()


### The function to test DRQN
def TEST(theta_test, testing_size=5000):

    with torch.no_grad():
        eval_net.eval()

        theta_state = np.random.rand(testing_size) - 0.5

        prev_resps_t = torch.full(
            (testing_size, 1), START_TOKEN, dtype=torch.long, device=device
        )
        test_hidden = eval_net.init_hidden(testing_size)

        item_id_history = np.empty((0, testing_size), dtype=np.int64)
        resp_history    = np.empty((0, testing_size), dtype=np.int64)
        dqn_step = np.zeros((1, 4))

        for i in range(test_length):
            action, test_hidden = Choose_Action_Test(
                prev_resps_t, test_hidden, item_id_history
            )
            response = RESPOND(item_bank[action], theta_test).astype(np.int64)

            item_id_history = np.concatenate((item_id_history, action[np.newaxis, :]))
            resp_history    = np.concatenate((resp_history,    response[np.newaxis, :]))

            theta_0 = np.zeros([1, testing_size])
            idx_full = np.sum(resp_history, axis=0) == resp_history.shape[0]
            idx_zero = np.sum(resp_history, axis=0) == 0
            idx_norm = np.bitwise_not(idx_full | idx_zero)
            theta_0[:, idx_full] = theta_state[idx_full] + (item_bank[:,1].max() - theta_state[idx_full]) / 2
            theta_0[:, idx_zero] = theta_state[idx_zero] + (item_bank[:,1].min() - theta_state[idx_zero]) / 2
            theta_0[:, idx_norm] = MLE_TEST(item_bank[item_id_history[:,idx_norm]], resp_history[:,idx_norm])

            if i == 0:
                theta = theta_0
            else:
                theta = np.concatenate((theta, theta_0))

            theta_state = theta_0[0]

            dqn_step = np.vstack([dqn_step, np.array([
                i+1,
                np.mean(theta_0 - theta_test),
                np.sqrt(np.mean((theta_0 - theta_test)**2)),
                np.mean(abs(theta_0 - theta_test)),
            ])])
            print("step {:g}, bias {:.3f}, rmse {:.3f}, mae {:.3f}".format(
                dqn_step[-1,0], dqn_step[-1,1], dqn_step[-1,2], dqn_step[-1,3]
            ))

            prev_resps_t = torch.from_numpy(response).long().unsqueeze(1).to(device)

        user_id  = np.repeat(np.arange(1, testing_size+1), test_length).reshape(-1, 1)
        step     = np.tile(np.arange(1, test_length+1), testing_size).reshape(-1, 1)
        item_id_out = (item_id_history + 1).transpose().reshape(-1, 1)
        resp_out    = resp_history.transpose().reshape(-1, 1)
        theta_est   = theta.transpose().reshape(-1, 1)
        bias        = (theta - theta_test).transpose().reshape(-1, 1)
        dqn_data = np.hstack([user_id, step, item_id_out, resp_out, theta_est, bias])
        dqn_data = pd.DataFrame(dqn_data).rename(
            columns={0:'userID', 1:'step', 2:'itemID', 3:'resp', 4:'theta_est', 5:'bias'}
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dqn_data.to_csv(RESULTS_DIR / f"records_{bank_type}_{bank_id}_DRQN_{prior}_gamma_{gamma}.csv", index=False)



### Hyperparameters
test_length  = 40
embed_dim    = 16
lstm_hidden  = 64
dropout_rate = 0.0

bank_type = 'uncor'
bank_id   = 1
prior     = "normal"  # "normal" for a standard normal distribution, "uniform" for a uniform distribution within [-3, 3]
gamma     = 0.1       # The optimal gamma value is in the t7 file name

### Load the item parameters
bank_dir = {
    "uncor": ROOT / "data" / "uncorrelated_banks",
    "cor":   ROOT / "data" / "correlated_banks",
}[bank_type]
item_bank = np.array(pd.read_csv(bank_dir / f"item_bank_{bank_type}_{bank_id}.csv")[['a','b','c']])
action_space = item_bank.shape[0]


### Create two recurrent Q-networks
eval_net   = DRQN(action_space, embed_dim, lstm_hidden, dropout_rate).to(device)
target_net = DRQN(action_space, embed_dim, lstm_hidden, dropout_rate).to(device)

### Initialize parameters
eval_net.initialize()
target_net.initialize()
target_net.load_state_dict(eval_net.state_dict())

### Start training phase (including validation)
TRAIN(gamma, prior)

### Start testing
theta_test = np.array(pd.read_csv(ROOT / "data" / "theta_true" / f"theta_true_{bank_id}.csv")['x'])
TEST(theta_test)


### Load the pre-trained model
eval_net = DRQN(action_space, embed_dim, lstm_hidden, dropout_rate).to(device)
eval_net = torch.load(MODEL_DIR / f"drqn_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.t7", weights_only=False)

### Start testing
theta_test = np.array(pd.read_csv(ROOT / "data" / "theta_true" / f"theta_true_{bank_id}.csv")['x'])
TEST(theta_test)
