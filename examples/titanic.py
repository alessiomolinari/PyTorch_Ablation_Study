import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pytorch_ablation.ablator import MaggyDataset, Ablator

from torch.autograd import Variable

# ## Data preprocessing

train_path = "data/titanic/titanic_train.csv"
test_path = "data/titanic/titanic_test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

test.head()

all_df = pd.concat([train, test])
all_df.head()

all_df['Embarked'] = all_df['Embarked'].astype(str)

all_df = all_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
dataset_features = list(all_df.columns.values)
# Encode labels

cat_cols = ['Pclass', 'Sex', 'SibSp', 'Parch']
for col in cat_cols:
    all_df[col] = LabelEncoder().fit_transform(all_df[col])

all_df = all_df.fillna(all_df.mean())

min_max_scaler = MinMaxScaler()
all_df = min_max_scaler.fit_transform(all_df)

# Split in train/test again

train_df = all_df[:train.shape[0]]
test_df = all_df[train.shape[0]:]


# ### Convert from pandas df to numpy array
# This is done for compatibility purposes with PyTorch
#
# In PyTorch, there are two important classes for data handling: Dataset and DataLoader
#
# **Dataset** is the parent class from which inherits your own dataset class. It keeps the data in a numpy format and
# overrides two methods \_\_getitem\_\_ and \_\_len\_\_ (len is optional) that will be called when you iterate
# through the dataset.
#
# **DataLoader** is the class that is used to iterate through the dataset and feed it to your model. You need to
# specify some parameters during creation like the dataset itself and hyperparameters such as batch_size, shuffle,
# num_workers, etc.

# PyTorch Data handling


class TitanicDataset(Dataset, MaggyDataset):
    def __init__(self, data):
        super(TitanicDataset).__init__()
        self.data = data

    # Optional override
    def __len__(self):
        return self.data.shape[0]

    # Compulsory override
    def __getitem__(self, idx):
        return self.data[idx]

    def ablate_feature(self, feature):
        # The user defines this function that explains the Ablator how to ablate a specific feature.
        # Not just columns can be ablated but even image channels and anything that is defined in the ablation space.
        global dataset_features
        idx = dataset_features.index(feature)
        self.data = np.delete(self.data, idx, axis=1)


# Instantiate dataset and dataloader
dataloader_kwargs = {"batch_size": 64, "shuffle": False}

dataset = TitanicDataset(train_df)
dataloader = DataLoader(dataset=dataset, **dataloader_kwargs)

model = nn.Sequential(nn.Linear(6, 64),
                      nn.ReLU(),
                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.Linear(64, 32),
                      nn.ReLU(),
                      nn.Linear(32, 32),
                      nn.ReLU(),
                      nn.Linear(32, 2),
                      nn.ReLU(),
                      nn.Linear(2, 1),
                      nn.Sigmoid()
                      )


def training_function(model, dataloader):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    training_history = []
    epochs = 5

    for epoch in range(epochs):

        epoch_loss = 0
        epoch_accuracy = 0

        for i, data in enumerate(dataloader):
            # print("load number {}".format(i))

            inputs = data[:, 1:]
            labels = data[:, 0]
            inputs, labels = Variable(inputs), Variable(labels)

            inputs = inputs.float()
            labels = labels.float().view(-1, 1)

            optimizer.zero_grad()
            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            epoch_loss += loss

            accuracy = ((y_pred > 0.5).float() == labels).float().mean()
            epoch_accuracy += accuracy

            loss.backward()
            optimizer.step()
        print("Epoch:{}, Loss:{}, Accuracy:{}".format(epoch,
                                                      epoch_loss.item(),
                                                      epoch_accuracy / dataloader.batch_size))
        training_history.append(epoch_loss.item() / len(train_df))
    print("Training complete\n")
    return loss


ablator = Ablator(model, dataset, dataset_features, dataloader_kwargs, training_function)
ablator.new_trial((6,), None, None)
ablator.new_trial((5,), None, "Sex")
ablator.new_trial((6,), 2, None, infer_activation=True)
ablator.new_trial((5,), 4, "Pclass")
ablator.new_trial((6,), [3, 4, 5, 6], None)

ablator.execute_trials()
