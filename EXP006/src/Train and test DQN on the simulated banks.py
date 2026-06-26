# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
EXP006_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = EXP006_DIR / "models"
RESULTS_DIR = EXP006_DIR / "results"


### If there is a GPU, set all training and testing to be done on the GPU, otherwise on the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### The function to simulate the examinee and generate the response under 3PLM
def RESPOND(item_para, theta, D=1):
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    p = (1 - c) / (1 + np.exp(-D * a * (theta - b))) + c
    resp = (np.random.rand(1) <= p).astype(int)
    return resp


### The function to calculate the Fisher item information under 3PLM
def FI(item_para, theta, D=1):
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    info = (
        D**2
        * a**2
        * (1 - c)
        / (c + np.exp(D * a * (theta - b)))
        / (1 + np.exp(-D * a * (theta - b))) ** 2
    )
    return info


### The function to calculate the negative log-posterior under a N(0, 1) prior
def NEG_LOG_POSTERIOR(theta, item_paras, resp, D=1):
    a = item_paras[:, 0]
    b = item_paras[:, 1]
    c = item_paras[:, 2]
    p = (1 - c) / (1 + np.exp(-D * a * (theta - b))) + c
    p = np.clip(p, 1e-12, 1 - 1e-12)
    log_likelihood = np.sum(resp * np.log(p) + (1 - resp) * np.log(1 - p))
    log_prior = -0.5 * ((theta - prior_mean) ** 2) / prior_var - 0.5 * np.log(
        2 * np.pi * prior_var
    )
    return float(-(log_likelihood + log_prior))


### The function to calculate the MAP estimate under 3PLM for the training phase
def MAP(item_paras, resp, D=1):
    theta = minimize_scalar(
        NEG_LOG_POSTERIOR,
        bounds=theta_bounds,
        method="bounded",
        args=(item_paras, resp, D),
    ).x
    return np.array(theta).reshape(
        1,
    )


### The function to parallelly deploy MAP estimation under 3PLM for validation and testing phases
def MAP_TEST(item_paras, resp, D=1):
    theta = np.zeros(resp.shape[1])
    for i in range(resp.shape[1]):
        theta[i] = minimize_scalar(
            NEG_LOG_POSTERIOR,
            bounds=theta_bounds,
            method="bounded",
            args=(item_paras[:, i, :], resp[:, i], D),
        ).x
    return np.expand_dims(theta, axis=0)


### Laplace approximation using the inverse curvature of the negative log-posterior at the MAP
def POSTERIOR_VAR(item_paras, theta, resp, D=1, step=1e-3):
    theta_lo, theta_hi = theta_bounds
    theta_center = float(np.clip(theta, theta_lo + step, theta_hi - step))
    f_minus = NEG_LOG_POSTERIOR(theta_center - step, item_paras, resp, D)
    f_center = NEG_LOG_POSTERIOR(theta_center, item_paras, resp, D)
    f_plus = NEG_LOG_POSTERIOR(theta_center + step, item_paras, resp, D)
    curvature = (f_plus - 2 * f_center + f_minus) / (step**2)
    curvature = max(curvature, min_curvature)
    return float(min(1.0 / curvature, max_var))


### Vectorized version over examinees for the validation and testing phases
### item_paras: shape (n_administered, N, 3), theta: shape (N,), resp: shape (n_administered, N)
def POSTERIOR_VAR_TEST(item_paras, theta, resp, D=1, step=1e-3):
    var = np.zeros(theta.shape[0])
    for j in range(theta.shape[0]):
        var[j] = POSTERIOR_VAR(
            item_paras[:, j, :],
            theta[j],
            resp[:, j],
            D=D,
            step=step,
        )
    return var


### The function to create a Q-network
class Net(nn.Module):
    def __init__(
        self,
        input_size,
        first_hidden,
        second_hidden,
        action_space,
        dropout_rate,
    ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.out = nn.Linear(second_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.dropout(self.fc2(x))
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


### The function to select items by Q-network with epsilon-greedy for training phase
def Choose_Action(item_id, state, epsilon):
    if np.random.randn() >= epsilon:
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        item_id = torch.from_numpy(item_id).to(device).long()
        action_value = eval_net(state)
        action_value[:, item_id] = torch.zeros(item_id.shape).to(device)
        action = torch.max(action_value, -1)[1].cpu().numpy()
    else:
        if any(item_id):
            action = (
                np.random.choice(np.delete(np.arange(action_space), item_id))
                .astype("int64")
                .reshape(
                    1,
                )
            )
        else:
            action = (
                np.random.choice(np.arange(action_space))
                .astype("int64")
                .reshape(
                    1,
                )
            )
    return action


### The function to select items by Q-network for validation and testing phases
def Choose_Action_Test(item_id, state):
    state = torch.FloatTensor(state.swapaxes(0, 1)).to(device)
    action_value = eval_net(state).detach().cpu().numpy()
    if item_id.shape[0] > 0:
        action_value[
            np.tile(np.arange(item_id.shape[1])[np.newaxis, :], (item_id.shape[0], 1)),
            item_id,
        ] = np.zeros(item_id.shape)
    action = action_value.argmax(axis=1)
    return action


### The function that limits the network parameters to positive, such that the output of Q-network (the expectation of cumulative information) is positive
def Apply_Positive_Constraint(model, min_value=0.0):
    for param in model.parameters():
        param.data = torch.clamp(param.data, min=min_value)


### The function to train Q-network including validation
def TRAIN(
    gamma,
    prior,
    memory_capacity=1000,
    epsilon=0.1,
    batch_size=128,
    q_network_iteration=40,
    learning_rate=1e-3,
    training_size=1000,
    validation_size=200,
    validation_interval=50,
):
    global best_valid

    loss_func = nn.MSELoss()
    eval_net.train()
    optimizer = optim.Adam(eval_net.parameters(), lr=learning_rate)

    memory = np.zeros((memory_capacity, input_size * 2 + 2))
    memory_counter = 0
    learn_step_counter = 0
    training_loss = 0
    training_reward = 0

    if prior == "normal":
        training_theta = np.random.randn(training_size)
    elif prior == "uniform":
        training_theta = np.random.uniform(-3, 3, training_size)

    for j in range(training_size):
        theta_hat = np.random.rand(1) - 0.5
        state = np.concatenate((theta_hat, [prior_var]))
        item_id = np.array([]).astype("int64")
        resp = np.array([]).astype("int64")

        for i in range(test_length):
            action = Choose_Action(item_id, state, epsilon)
            item_id = np.concatenate((item_id, action))
            resp = np.concatenate((resp, RESPOND(item_bank[action], training_theta[j])))
            reward = FI(item_bank[action,], training_theta[j])
            next_theta = MAP(item_bank[item_id,], resp)
            next_var = POSTERIOR_VAR(item_bank[item_id], next_theta[0], resp)
            next_state = np.concatenate((next_theta, [next_var]))

            memory[memory_counter % memory_capacity, :] = np.hstack(
                (state, action, reward, next_state)
            )
            memory_counter += 1
            state = next_state

            if memory_counter >= batch_size:
                batch_memory = memory[
                    np.random.choice(min(memory_counter, memory_capacity), batch_size),
                    :,
                ]
                batch_state = torch.FloatTensor(batch_memory[:, :input_size]).to(device)
                batch_action = torch.LongTensor(
                    batch_memory[:, input_size : input_size + 1].astype(int)
                ).to(device)
                batch_reward = torch.FloatTensor(
                    batch_memory[:, input_size + 1 : input_size + 2]
                ).to(device)
                batch_next_state = torch.FloatTensor(batch_memory[:, -input_size:]).to(
                    device
                )

                q_eval = eval_net(batch_state).gather(1, batch_action)
                q_next = target_net(batch_next_state).detach()
                if i == test_length - 1:
                    q_target = batch_reward
                else:
                    q_target = batch_reward + gamma * q_next.max(1)[0].view(
                        batch_size, 1
                    )
                loss = loss_func(q_eval, q_target)

                optimizer.zero_grad()
                loss.backward()
                Apply_Positive_Constraint(eval_net)
                optimizer.step()

                training_reward += reward[0]
                training_loss += loss.item()
                learn_step_counter += 1

                if learn_step_counter % q_network_iteration == 0:
                    target_net.load_state_dict(eval_net.state_dict())

        ### Validation ###
        if (j + 1) % validation_interval == 0:
            eval_net.eval()
            valid_loss = 0
            valid_reward = 0
            valid_bias = np.zeros((test_length, validation_size))
            valid_theta = np.random.choice(training_theta, validation_size)

            theta_hat0 = (np.random.rand(validation_size) - 0.5)[np.newaxis, :]
            var0 = np.full((1, validation_size), prior_var)
            state = np.vstack((theta_hat0, var0))
            item_id = np.array([])

            for i in range(test_length):
                action = Choose_Action_Test(item_id, state)
                reward = FI(item_bank[action,], valid_theta)[:, np.newaxis]
                if i == 0:
                    item_id = action[np.newaxis, :]
                    resp = RESPOND(item_bank[action,], valid_theta)[np.newaxis, :]
                else:
                    item_id = np.concatenate((item_id, action[np.newaxis, :]))
                    resp = np.concatenate(
                        (resp, RESPOND(item_bank[action,], valid_theta)[np.newaxis, :])
                    )

                theta_0 = np.squeeze(MAP_TEST(item_bank[item_id], resp))

                q_eval = eval_net(
                    torch.FloatTensor(np.transpose(state)).to(device)
                ).gather(
                    1,
                    torch.LongTensor(
                        action[
                            :,
                            np.newaxis,
                        ]
                    ).to(device),
                )
                var_0 = POSTERIOR_VAR_TEST(item_bank[item_id], theta_0, resp)
                state = np.vstack((theta_0[np.newaxis, :], var_0[np.newaxis, :]))
                q_next = target_net(
                    torch.FloatTensor(np.transpose(state)).to(device)
                ).detach()
                if i == test_length - 1:
                    q_target = torch.FloatTensor(reward).to(device)
                else:
                    q_target = (
                        torch.FloatTensor(reward).to(device) + gamma * q_next.max()
                    )
                valid_loss += loss_func(q_eval, q_target).item()
                valid_reward += np.mean(reward)
                valid_bias[i] = theta_0 - valid_theta

            step_valid = np.transpose(
                np.vstack(
                    (
                        np.arange(1, test_length + 1),
                        np.mean(valid_bias, axis=1),
                        np.sqrt(np.mean(valid_bias**2, axis=1)),
                        np.mean(abs(valid_bias), axis=1),
                    )
                )
            )

            print(
                "subject: {}, loss: {:.3f}, reward: {:.3f}\n\n{}\n".format(
                    j + 1,
                    valid_loss / validation_size,
                    valid_reward,
                    step_valid,
                )
            )

            result_valid = np.mean(step_valid[6:, 1:], axis=0)

            model_path = (
                MODEL_DIR
                / f"dqn_maphess_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.t7"
            )
            try:
                best_valid
            except NameError:
                best_valid = result_valid
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(eval_net, model_path)

            if (abs(result_valid[0]) < abs(best_valid[0])) & np.sum(
                result_valid[1:] < best_valid[1:]
            ) == 2:
                best_valid = result_valid
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(eval_net, model_path)

            eval_net.train()


### The function to test Q-network
def TEST(theta_test, testing_size=5000):
    with torch.no_grad():
        eval_net.eval()

        theta_hat0 = (np.random.rand(testing_size) - 0.5)[np.newaxis, :]
        var0 = np.full((1, testing_size), prior_var)
        state = np.vstack((theta_hat0, var0))
        item_id = np.array([])
        dqn_step = np.zeros((1, 4))

        for i in range(test_length):
            action = Choose_Action_Test(item_id, state)
            if i == 0:
                item_id = action[np.newaxis, :]
                resp = RESPOND(item_bank[action,], theta_test)[np.newaxis, :]
            else:
                item_id = np.concatenate((item_id, action[np.newaxis, :]))
                resp = np.concatenate(
                    (resp, RESPOND(item_bank[action,], theta_test)[np.newaxis, :])
                )

            theta_0 = MAP_TEST(item_bank[item_id], resp)

            if i == 0:
                theta = theta_0
            else:
                theta = np.concatenate((theta, theta_0))

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
                    dqn_step[-1, 0],
                    dqn_step[-1, 1],
                    dqn_step[-1, 2],
                    dqn_step[-1, 3],
                )
            )

            var_0 = POSTERIOR_VAR_TEST(item_bank[item_id], theta_0[0], resp)
            state = np.vstack((theta_0, var_0[np.newaxis, :]))

        user_id = np.repeat(np.arange(1, testing_size + 1), test_length).reshape(-1, 1)
        step = np.tile(np.arange(1, test_length + 1), testing_size).reshape(-1, 1)
        item_id = (item_id + 1).transpose().reshape(-1, 1)
        resp = resp.transpose().reshape(-1, 1)
        theta_est = theta.transpose().reshape(-1, 1)
        bias = (theta - theta_test).transpose().reshape(-1, 1)
        dqn_data = np.hstack([user_id, step, item_id, resp, theta_est, bias])
        dqn_data = pd.DataFrame(dqn_data).rename(
            columns={
                0: "userID",
                1: "step",
                2: "itemID",
                3: "resp",
                4: "theta_est",
                5: "bias",
            }
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dqn_data.to_csv(
            RESULTS_DIR
            / f"records_{bank_type}_{bank_id}_DQNmaphess_{prior}_gamma_{gamma}.csv",
            index=False,
        )


### Hyperparameters
test_length = 40

input_size = 2  # [theta_map, var_map]
theta_bounds = (-4, 4)
prior_mean = 0.0
prior_var = 1.0
max_var = 1.0
min_curvature = 1e-6
first_hidden = 50
second_hidden = 30
dropout_rate = 0


bank_type = "uncor"
bank_id = 1
prior = "normal"  # theta generation prior, not the estimation prior
gamma = 0.1  # The optimal gamma value is in the t7 file name

### Load the item parameters
bank_dir = {
    "uncor": ROOT / "data" / "uncorrelated_banks",
    "cor": ROOT / "data" / "correlated_banks",
}[bank_type]
item_bank = np.array(
    pd.read_csv(bank_dir / f"item_bank_{bank_type}_{bank_id}.csv")[["a", "b", "c"]]
)
action_space = item_bank.shape[0]


### Create two Q-networks
eval_net = Net(input_size, first_hidden, second_hidden, action_space, dropout_rate).to(
    device
)
target_net = Net(
    input_size, first_hidden, second_hidden, action_space, dropout_rate
).to(device)

### Initialize parameters in Q-networks
eval_net.initialize()
target_net.initialize()

### Start training phase (including validation)
TRAIN(gamma, prior)

### Start testing
theta_test = np.array(
    pd.read_csv(ROOT / "data" / "theta_true" / f"theta_true_{bank_id}.csv")["x"]
)
TEST(theta_test)


### Load the pre-trained model
eval_net = Net(input_size, first_hidden, second_hidden, action_space, dropout_rate).to(
    device
)
eval_net = torch.load(
    MODEL_DIR / f"dqn_maphess_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.t7"
)

### Start testing
theta_test = np.array(
    pd.read_csv(ROOT / "data" / "theta_true" / f"theta_true_{bank_id}.csv")["x"]
)
TEST(theta_test)
