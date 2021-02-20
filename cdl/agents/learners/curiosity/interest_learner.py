# import numpy as np
# import random
# import copy
# import math
# import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from lems.utils.objects import MoveConfig
# from lems.utils.io import save_json, save_str, load_json, load_str
# from lems.utils.serializer import Serializer, getReference
# # from lems.utils.debug import timethissub
# from lems.utils.maths import uniformSampling

# from lems.agents.learners.model.model_learner import ModelLearner
# from lems.data.dataset import Dataset

# from ...data.interest_model_manager import InterestModelManager
# from ...data.model_manager import ModelManager
# from ...models.interest_model import InterestModel
# from ...maps.map import Map
# from .mods import Mod

from dino.agents.learners.model.model import ModelLearner

from dino.utils.maths import uniformSampling

from cdl.agents.tools.managers.interest_model_manager import InterestModelManager
from cdl.agents.tools.managers.adaptive_model_manager import AdaptiveModelManager
from cdl.agents.tools.models.interest_model import InterestModel


class InterestLearner(ModelLearner):
    """Complete learning agent with an interest model and mods."""
    MODEL_CLASS = InterestModel

    def __init__(self, host, dataset=None, mods=[], performer=None, planner=None, options={}):
        """
        dataset Dataset: dataset of the agent
        strat StrategyV4 list: list of learning strategies available to the agent
        mods Mod list: list of available mods (used to determine wich task/goal/strategy to explore
        interestModel InterestModelManager: interest model used by the agent to guide its learning process
        env EnvironmentV4: host of the experiment
        """
        # options['dataset'] = dict(options.get('dataset', {}), **{'model':})
        super().__init__(host, dataset=dataset, performer=performer,
                         planner=planner, options=options)
        self.interestModel = InterestModelManager(self)
        self.mods = []
        for mod in mods:
            self.addMod(mod)

        self.adaptiveModels = True
        self.adaptiveModelManager = None

        # creates semantic/metric map
        # self.semmap = Map(environment)
        # self.planner.semmap = self.semmap

        self.configHistory = []

        # self.progresses = []
        # self.goal_progresses = []

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['mods', 'adaptiveModels']))
        return dict_
    
    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        self.adaptiveModels = dict_.get('adaptiveModels', self.adaptiveModels)

    # @classmethod
    # def _deserialize(cls, dict_, environment, dataset=None, options={}, obj=None):
    #     mods = [Mod.deserialize(dictMod) for dictMod in dict_.get('mods', [])]
    #     obj = obj if obj else cls(
    #         environment, dataset, mods, options=dict_.get('options', {}))
    #     obj = ModelLearner._deserialize(
    #         dict_, environment, dataset=dataset, options=options, obj=obj)

    #     # Strategies
    #     # from lems.utils.loaders import DataManager
    #     # for s in dict_.get('testStrategies', []):
    #     #     cls = DataManager.loadType(s['path'], s['type'])
    #     #     obj.testStrategies.add(cls.deserialize(s, obj))
    #     # for s in dict_.get('trainStrategies', []):
    #     #     cls = DataManager.loadType(s['path'], s['type'])
    #     #     obj.trainStrategies.add(cls.deserialize(s, obj))

    #     return obj

    def addMod(self, mod):
        mod.agent = self
        self.mods.append(mod)

    def sampleMod(self):
        """Choose mod stochastically."""
        return self.mods[uniformSampling([m.prob for m in self.mods])]

    def _preEpisode(self):
        # Choose learning strategy randomly
        mod = self.sampleMod()
        # self.logger.debug('Mod {} selected'.format(getReference(mod)), 'STRAT')

        # Select task, strategy and goal outcome according to chosen mod
        config = mod.sample(self.environment.world.currentContext(self.dataset))
        config.iteration = self.iteration
        self.configHistory.append(config)
        # self.logger.debug('Strategy {} and goal {} selected: {}'
        #                   .format(getReference(config.strategy), getReference(config.goal),
        #                           getReference(config.sampling)), 'STRAT')

        # Make sure strategy chosen is available
        # while not self.strategies[strat].available(task):
        #     task, strat, goal = self.sampleStrategyGoal(mod)
        self.logger.debug(f'Iteration {self.iteration}: {mod} chose {config}', tag='strategy')
        return config

    def _addEvent(self, event, config, cost=1.):
        self.interestModel.addEvent(event, config.strategy)
        if self.adaptiveModels:
            if not self.adaptiveModelManager:
                self.adaptiveModelManager = AdaptiveModelManager(self)
            self.adaptiveModelManager.addEvent(event)
