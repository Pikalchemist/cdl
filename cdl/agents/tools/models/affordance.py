from collections import deque
import random
import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable

from .interest_model import InterestModel

from dino.data.abstract import AMT
from dino.data.space import SpaceKind


class Affordance(InterestModel):
    def __init__(self, dataset, actionSpace, outcomeSpace, contextSpace=[],
                 restrictionIds=None, model=None, register=True, metaData={}):

        self.amt = AMT()
        self.abstractOutcomeSpace = AMT(self.amt)
        self.abstractOutcomeSpace.abstractSpace(
            outcomeSpace, appendToChildren=True)

        self.abstractActionSpace = AMT(self.amt)
        self.abstractActionSpace.abstractSpace(
            actionSpace, abstractNewElements=False, appendToChildren=True)

        contextSpace = contextSpace.convertTo(dataset, kind=SpaceKind.PRE)
        self.abstractContextSpace = AMT(self.amt)
        if contextSpace:
            self.abstractContextSpace.abstractSpace(
                contextSpace, abstractNewElements=False, appendToChildren=True)

        # self.assignables = []
        # self.nonAssignables = []

        # if assignables:
        #     element = list(self.amt.assignableElements().keys())[0]
        #     for assignable in assignables:
        #         element.assignable(assignable)

        outcomeSpace = [self.abstractOutcomeSpace.get(dataset)]
        actionSpace = [self.abstractActionSpace.get(dataset)]
        contextSpace = [self.abstractContextSpace.get(dataset)]

        super().__init__(dataset, actionSpace, outcomeSpace, contextSpace, restrictionIds, model,
                         register, metaData)

        # Visual classifier

    # @classmethod
    # def createFromOther(cls, affordance, restrictionIds=None, register=True):
    #     return cls(affordance.dataset, )
    
    def metaData(self):
        return {}
    
    @property
    def semanticMap(self):
        return self.dataset.learner.environment
    
    def updateSpaces(self):
        self.actionSpace = self.dataset.multiColSpace(
            [self.abstractActionSpace.get(self.dataset)])
        self.outcomeSpace = self.dataset.multiColSpace(
            [self.abstractOutcomeSpace.get(self.dataset)])
        self.contextSpace = self.dataset.multiColSpace(
            [self.abstractContextSpace.get(self.dataset)])

        self.actionContextSpace = self.dataset.multiColSpace(
            [self.actionSpace, self.contextSpace], weight=0.5)
        self.outcomeContextSpace = self.dataset.multiColSpace(
            [self.outcomeSpace, self.contextSpace], weight=0.5)

        self.invalidateCompetences()
    
    def mainEntity(self):
        entities = self.abstractOutcomeSpace.abstractedEntities()
        assert(entities == 1, 'Only 1 abstract entity may exist for the outcome space!')
        return list(entities)[0]

    def requiredProperties(self):
        return self.amt.abstractedEntityProperties().get(self.mainEntity(), set())

    def compatibleEntity(self, entity):
        print(self.requiredProperties())
        print(entity.properties())
        for required in self.requiredProperties():
            if not entity.property(required.name()):
                return False
        return True
    
    def addAssignable(self, entity):
        self.mainEntity().assignable(entity)
        self.updateSpaces()


# class VisualClassifier(nn.Module):
#     def __init__(self, visualSpace):
#         self.visualSpace = visualSpace
#         self.visualSpace._validate()

#         N = 16

#         self.fc1 = nn.Linear(self.visualSpace.dim, N)
#         self.fc2 = nn.Linear(N, N // 2)
#         self.fc3 = nn.Linear(N // 2, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.Sigmoid(self.fc3(x))
#         return x


# class VisualClassifierTrainer(object):
#     LEARNING_RATE = 0.001
#     MEMORY_SIZE = 1000000
#     BATCH_SIZE = 20

#     def __init__(self, classifier):
#         self.model = classifier

#         self.memory = (deque(maxlen=self.MEMORY_SIZE),
#                        deque(maxlen=self.MEMORY_SIZE))
#         # Double deque: (deque for negative results, deque for positive results)

#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(
#             self.model.parameters(), lr=self.LEARNING_RATE)

#     def remember(self, object, state, applicable):
#         state = state
#         self.memory[applicable].append(
#             (object, torch.from_numpy(state.astype(np.float32))))

#     def applicable(self, object, applicable):
#         changing = []
#         for event in self.memory[1-applicable]:
#             if event[0] == object:
#                 changing.append(event)
#         for event in changing:
#             self.memory[1-applicable].remove(event)
#             self.memory[applicable].append(event)

#     def train(self):
#         if len(self.memory) < self.BATCH_SIZE:
#             return
#         batch = random.sample(self.memory, self.BATCH_SIZE)
#         for state, action, reward, state_next, terminal in batch:
#             q_update = reward
#             if not terminal:
#                 # print(state_next)
#                 q_update = (reward + self.GAMMA *
#                             self.model(state_next).max(0)[0])
#             q_values = self.model(state)
#             # print(action)
#             # print('---')
#             # print(q_values)
#             # print(state)
#             # print(self.model(state_next).max(0)[0])
#             q_values[action] = q_update
#             # print(q_values)

#             # self.model.fit(state, q_values, verbose=0)

#             vstate, vq_values = Variable(state), Variable(q_values)

#             self.optimizer.zero_grad()
#             outputs = self.model(vstate)
#             loss = self.criterion(outputs, vq_values)
#             loss.backward()
#             self.optimizer.step()

#             print('! {}  {}  {}  {}  {}'.format(loss, reward, outputs.data.numpy(
#             ), self.model(vstate).data.numpy(), q_values.data.numpy()))
