import random
import numpy as np

from contextlib import contextmanager
from collections import deque

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable

from sklearn.neural_network import MLPClassifier

from .interest_model import InterestModel

from dinos.data.abstract import AMT
from dinos.data.space import SpaceKind


class Affordance(InterestModel):
    COMPETENCE_MARGIN = 0.05
    COMPETENCE_MIN = 0.2
    MIN_NON_ZERO = 0.001

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
        
        if metaData:
            for assignable in metaData.get('assignables', []):
                self.addAssignable(assignable)

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
        self.visualPredictor = VisualClassifier()
        for entity in self.assignables():
            self.addVisualSample(entity, True)

    # @classmethod
    # def createFromOther(cls, affordance, restrictionIds=None, register=True):
    #     return cls(affordance.dataset, )

    @classmethod
    def adaptiveEpisode(cls, adaptiveModelManager, events):
        print('HEYHEYHEYHEY')
        ids = [event.iteration for event in events]
        for affordance in adaptiveModelManager.dataset.enabledModels():
            entities = affordance.interactedEntities(events)
            if entities:
                affordance.testMatchingEntities(entities, ids=ids)
    
    def justCreated(self):
        print('CREATED!!!')
        # self.testMatchingEntities()
    
    def metaData(self):
        return {'assignables': self.assignables()}
    
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
    
    def mainAbstractEntity(self):
        entities = self.abstractOutcomeSpace.abstractedEntities()
        assert(entities == 1, 'Only 1 abstract entity may exist for the outcome space!')
        return list(entities)[0]

    def addAssignable(self, entity):
        self.mainAbstractEntity().assignable(entity)
        self.updateSpaces()

    def removeAssignable(self, entity):
        self.mainAbstractEntity().unassignable(entity)
        self.updateSpaces()

    def assignables(self):
        return self.mainAbstractEntity().assignables

    def requiredProperties(self):
        return self.amt.abstractedEntityProperties().get(self.mainAbstractEntity(), set())

    def compatibleEntity(self, entity):
        for required in self.requiredProperties():
            if not entity.propertyItem(required.name()):
                return False
        return True

    def compatibleEntities(self, excludeAlreadyIncluded=False):
        return [entity for entity in self.dataset.learner.environment.world.cascadingChildren()
                if (not excludeAlreadyIncluded or entity not in self.assignables()) and self.compatibleEntity(entity)]
    
    def applicableEntity(self, entity):
        return self.visualPredictor.predict(entity.observeVisualProperties())

    def applicableEntities(self):
        return [entity for entity in self.compatibleEntities() if self.applicableEntity(entity)]
    
    def addVisualSample(self, entity, applicable):
        self.visualPredictor.addSample(entity.observeVisualProperties(), applicable)

    def competenceForEntity(self, entity, ids=None, precise=True):
        assert(self.compatibleEntity(entity), f'{entity} is not compatible! (missing required properties)')

        with self.mainAbstractEntity().assign(entity):
            actionSpace = self.abstractActionSpace.get(self.dataset)
            outcomeSpace = self.abstractOutcomeSpace.get(self.dataset)
            contextSpace = self.abstractContextSpace.get(self.dataset)
        ids = self.findSharedIds(actionSpace, outcomeSpace, contextSpace, restrictionIds=ids)
        c = self.competenceData(actionSpace.getPoint(ids, toSpace=self.actionSpace),
                                outcomeSpace.getPoint(ids, toSpace=self.outcomeSpace),
                                contextSpace.getPoint(ids, toSpace=self.contextSpace), precise=precise)

        return c
    
    def testMatchingEntities(self, entities=None):
        if not entities:
            entities = self.compatibleEntities(excludeAlreadyIncluded=True)

        left = list(entities)
        for entity in entities:
            left.remove(entity)
            newAffordance = self.testMatchingEntity(entity)
            if newAffordance != self:
                return newAffordance.testMatchingEntities(left)

        return self

    def testMatchingEntity(self, entity, ids=None):
        if self.tryAddingEntity(entity, ids=ids):
            return self
        
        # Try modifying context to match entity
        if not self.dataset.learner.adaptiveModelManager:
            return self

        def externalCriterium(newAffordance):
            if newAffordance.tryAddingEntity(entity, ids=ids, resetAfter=False):
                return -self.COMPETENCE_MARGIN
            return 1

        return self.dataset.learner.adaptiveModelManager.updateModel(self, externalCriterium=externalCriterium)

    def tryAddingEntity(self, entity, ids=None, resetAfter=True):
        success = self._tryAddingEntityWithoutAddingSample(entity, ids=ids, resetAfter=resetAfter)
        self.addVisualSample(entity, success)
        return success
    
    def _tryAddingEntityWithoutAddingSample(self, entity, ids=None, resetAfter=True):
        if entity in self.assignables():
            return True

        ce = self.competenceForEntity(entity, precise=True, ids=ids)
        if ce < self.COMPETENCE_MIN:
            return False

        c = self.competence(precise=True)

        if ce < c - self.COMPETENCE_MARGIN:
            return False
        
        if ids is not None:
            ce = self.competenceForEntity(entity, precise=True)
            if ce < c - self.COMPETENCE_MARGIN:
                return False

        self.addAssignable(entity)
    
        nc = self.competence(precise=True)
        if nc < c - self.COMPETENCE_MARGIN:
            if resetAfter:
                self.removeAssignable(entity)
            return False
        
        return True
    
    def interactedEntities(self, events):
        entities = []

        for entity in self.compatibleEntities():
            with self.mainAbstractEntity().assign(entity):
                outcomeSpace = self.abstractOutcomeSpace.get(self.dataset)
            for event in events:
                d = event.outcomes.projection(outcomeSpace)
                if d and d.norm1() > self.MIN_NON_ZERO:
                    entities.append(entity)
                    break

        return entities

    @contextmanager
    def allowEntity(self, entity):
        try:
            with self.mainAbstractEntity().assign(entity):
                actionSpace = self.abstractActionSpace.get(self.dataset)
                outcomeSpace = self.abstractOutcomeSpace.get(self.dataset)
                contextSpace = self.abstractContextSpace.get(self.dataset)

                self.actionSpace.allowSimilarRows([actionSpace])
                self.outcomeSpace.allowSimilarRows([outcomeSpace])
                self.contextSpace.allowSimilarRows([contextSpace])
            yield self
        finally:
            self.actionSpace.resetSimilarRows()
            self.outcomeSpace.resetSimilarRows()
            self.contextSpace.resetSimilarRows()
    
    def forward(self, action, context=None, contextColumns=None, ignoreFirst=False, entity=None):
        if not entity:
            return super().forward(action, context, contextColumns=contextColumns, ignoreFirst=ignoreFirst, entity=entity)
        with self.allowEntity(entity):
            if context:
                context = context.projection(self.contextSpace, entity=entity)
            return super().forward(action, context, contextColumns=contextColumns, ignoreFirst=ignoreFirst, entity=entity)
    
    def inverse(self, goal, context=None, adaptContext=False, contextColumns=None, entity=None):
        if not entity:
            return super().inverse(goal, context, adaptContext=adaptContext, contextColumns=contextColumns, entity=entity)
        with self.allowEntity(entity):
            if context:
                context = context.projection(self.contextSpace, entity=entity)
            return super().inverse(goal, context, adaptContext=adaptContext, contextColumns=contextColumns, entity=entity)


class VisualClassifier(object):
    def __init__(self):
        self.model = MLPClassifier(random_state=1, max_iter=300)

        self.data = []
        self.applicable = []

    def addSample(self, data, applicable):
        self.data.append(data)
        self.applicable.append(applicable)
        self.train()

    def train(self):
        self.model.fit(np.array(self.data), np.array(self.applicable))

    def predict(self, visualValues):
        return self.model.predict(np.array([visualValues]))[0]


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
