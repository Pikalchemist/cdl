import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy
import random

import pickle

from exlab.modular.module import Module
from exlab.utils.io import parameter

from dino.data.data import Data
from dino.data.region import SpaceRegion

from dino.utils.move import MoveConfig
from dino.utils.maths import uniformRowSampling, mixedSort

# from collections import namedtuple

# from lems.utils.io import getVisual, save_json, save_str, load_json, load_str, save_raw, load_raw, ptstr, plotData
# from lems.utils.serializer import Serializer
# from lems.utils.maths import uniformSampling, uniformRowSampling, mixedSort
# from lems.utils.objects import MoveConfig
# from lems.utils.logging import DataEventHistory, DataEventKind


# from lems.data.data import SingleAction, Action, ActionList, SingleObservation, Observation, Goal, SingleData, Data, InteractionEvent
# from lems.data.space import DataSpace, SpaceKind
# from lems.data.region import SpaceRegion

# RegionSampling = namedtuple('RegionSampling', ['region', 'model', 'strategy'])


class InterestModelManager(Module):
    """Model of interest for the intrinsic motivated learner."""

    def __init__(self, learner, options={}):
        """
        dataset Dataset: dataset of the learner
        options dict: contain parameters of the model
        """
        super().__init__("InterestModelManager", learner, 'interest')
        # Different keys of options
        # - numberRegionSelection: maximum number of best regions considered when choosing with intrinsic motivation
        # - around: radius of the ball used to sample around a specific point
        # - costs: list containing the cost of each available strategy for the learner
        # - ## contains other options described in InterestRegion class documentation
        self.dataset = learner.dataset
        self.method = SpaceRegion.POINT_BASED
        # self.dataset.addChildModule(self)
        self.options = {
            'around': 0.05,
            'numberRegionSelection': 10,
            'sampleContextRatio': 0.
        }
        self.options.update(options)

        self.maps = []
        self.debutSampling = []
        self.spaces = []

    def __repr__(self):
        return "InterestModelManager -> " + str(self.maps)
    
    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(self, ['options']))
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.get('agent'), dict_.get('options', {}))
        return super()._deserialize(dict_, serializer, obj)

    def createInterestMap(self, model, strategy):
        self.logger.info(
            f'Creating interestMap for model {model}, strategy {strategy}', tag='interest')
        mp = InterestRegion(model.outcomeSpace, self.options, contextSpace=model.contextSpace,
                            manager=self, model=model, strategy=strategy)
        model.interestMaps[strategy] = mp
        self.maps.append(mp)
        return mp

    def addEvent(self, event, strategy):
        actions = event.actions
        outcomes = event.outcomes
        context = event.context if event.context else Data()
        models = [model for model in self.dataset.enabledModels()
                  if model.isCoveredByOutcomeSpaces(outcomes.space)
                  and model.isCoveredByContextSpaces(context.space)
                  and not np.all(outcomes.projection(model.outcomeSpace).npPlain() == 0)]

        # competences = [model.goalCompetence(outcomes, context) for model in models]
        self.dataset.addEvent(event)

        for model in models:
            if model.isCoveredByActionSpaces(actions.space):
                action = actions.projection(model.actionSpace)
            elif model.isCoveredByActionSpaces(event.primitiveActions.space):
                action = event.primitiveActions.projection(model.actionSpace)
            else:
                action = outcomes.projection(model.actionSpace)
            outcome = outcomes.projection(model.outcomeSpace)
            # print(f'OUTCOME {outcome}')
            # contextColumns = model.contextColumns(None, action, context.projection(model.contextSpace))
            predictedOutcome = model.forward(action, context=context)[0]
            closeIds = model.lastCloseIds
            if predictedOutcome and model.outcomeSpace.number > 15:
                # print(f'{predictedOutcome} {outcome}')
                # print(f'predicted {predictedOutcome.plain()} and got {outcome.plain()}')
                distance = min(predictedOutcome.distanceTo(outcome) / model.maxDistance(outcome), 1.)# * 10.
                # print(f'INTEREST ERROR: {model} {distance}')
                if distance > 0.01:
                    self.logger.info(f'ERROR {distance:.4f} for action {action} predictedOutcome {predictedOutcome} =? {outcome} model {model}')

                if strategy not in model.interestMaps:
                    self.createInterestMap(model, strategy)
                model.interestMaps[strategy].addPoint(event.iteration, outcomes, context, distance, closeIds=closeIds)
        
        for model in models:
            model.pointAdded(event)

        # progresses = [self.computeProgress(previousCompetence, model.goalCompetence(outcomes, context))
        #               for model, previousCompetence in zip(models, competences)]
        # for model, progress in zip(models, progresses):
        #     if strategy not in model.interestMaps:
        #         self.createInterestMap(model, strategy)
        #     model.interestMaps[strategy].addPoint(
        #         outcomes.extends(context), progress)
        #     model.pointAdded(event, progress)

    # def computeProgress(self, from_, to):
    #     return (to - from_) * (max(to, from_) ** 2)

    # def get_interest_tree(self, space):
    #     return first([it for it in self.maps if it.space == space])

    # def interesttree_to_space(self, interesttree):
    #     return self.spaces[interesttree]
    #
    # def space_to_interesttree(self, space):
    #     if space not in self.spaces:
    #         return None
    #     return self.spaces.index(space)

    def modelsWithInterestMaps(self, strategiesAvailable=None):
        if strategiesAvailable:
            return [m for m in self.dataset.models if m.enabled and strategiesAvailable.intersection(set(m.interestMaps.keys()))]
        return [m for m in self.dataset.models if m.enabled and m.interestMaps]

    # Samples
    def sampleRandomAction(self, strategiesAvailable=[], context=None, noContextGoal=False):
        """Sample action."""
        if not strategiesAvailable:
            models = self.modelsWithInterestMaps()
            if models:
                strategiesAvailable = set([strategy for model in models for strategy, _ in model.interestMaps.items()])
        strategy = random.choice(list(strategiesAvailable))

        if not noContextGoal:
            pass

        return MoveConfig(strategy=strategy, sampling="random space, random action")

    def sampleRandomPoint(self, strategiesAvailable=None, context=None):
        """Sample one point at random in the regions."""
        region, changeContext = self.sampleRandomRegion(strategiesAvailable=strategiesAvailable, context=context)
        if region is None:
            return self.sampleRandomAction(strategiesAvailable=strategiesAvailable, context=context, noContextGoal=True)

        goal, goalContext = region.sampleRandomPoint(context, changeContext)

        return MoveConfig(model=region.model, strategy=region.strategy, goal=goal, goalContext=goalContext,
                          sampling="Random region, random goal")

    def sampleGoodPoint(self, strategiesAvailable=[], context=None):
        """Sample one point at random in one of the best regions."""
        region, changeContext = self.sampleBestRegion(strategiesAvailable=strategiesAvailable, context=context)
        if region is None:
            return self.sampleRandomAction(strategiesAvailable=strategiesAvailable, context=context, noContextGoal=True)

        goal, goalContext = region.sampleGoodPoint(context, changeContext)
        goalContext = self.dataset.controlContext(goalContext, context)

        return MoveConfig(model=region.model, strategy=region.strategy, goal=goal, goalContext=goalContext,
                          sampling=f"best region in space {region.explorableSpace} with an interest of {region.evaluation:.4f}, random goal")

    def sampleBestPoint(self, strategiesAvailable=[], context=None):
        """Sample one point around the best point in one of the best regions."""
        region, changeContext = self.sampleBestRegion(strategiesAvailable=strategiesAvailable, context=context)
        if region is None:
            return self.sampleRandomAction(strategiesAvailable=strategiesAvailable, context=context, noContextGoal=True)
        
        goal, goalContext = region.sampleBestPoint(context, changeContext)
        goalContext = self.dataset.controlContext(goalContext, context)

        return MoveConfig(model=region.model, strategy=region.strategy, goal=goal, goalContext=goalContext,
                          sampling=f"best region in space {region.explorableSpace} with an interest of {region.evaluation:.4f}, goal around best point")

    def chooseToChangeContext(self, bestChangedContext, bestCurrentContext):
        if bestCurrentContext > bestChangedContext * 4:
            return False
        if bestChangedContext > bestCurrentContext * 4:
            return True
        return random.uniform(0, bestChangedContext + bestCurrentContext) < bestChangedContext
    
    def changeContextDecision(self, contextProbability=0.5):
        return random.uniform(0, 1) < contextProbability

    def sampleBestRegion(self, strategiesAvailable=[], context=None):
        return self.sampleRegion(strategiesAvailable, context, best=True)

    def sampleRandomRegion(self, strategiesAvailable=[], context=None):
        return self.sampleRegion(strategiesAvailable, context, best=False)

    def sampleRegion(self, strategiesAvailable=[], context=None, best=False):
        # List all available regions
        regions = []
        for model in self.dataset.enabledModels():
            for strategy, mp in model.interestMaps.items():
                if not strategiesAvailable or strategy in strategiesAvailable:
                    for region in mp.regions:
                        if region.evaluation != 0.0:
                            # print(f'{region.model}: {region.evaluation}')
                            regions.append(region)

        # Check if region context may be controlled
        regionsChangedContext = list(regions)  # [region for region in regions if region.controllableContext()]
        bestChangedContext = np.max([np.abs(region.evaluation) for region in regionsChangedContext]) if regionsChangedContext else 0

        # Filter by context
        # if not sampleContext and context:
        regionsCurrentContext = []
        if context:
            regionsCurrentContext = [region for region in regions if region.nearContext(context)]
        bestCurrentContext = np.max([np.abs(region.evaluation) for region in regionsCurrentContext]) if regionsCurrentContext else 0

        if not regionsChangedContext and not regionsCurrentContext:
            return None, False

        if best:
            changeContext = self.chooseToChangeContext(bestChangedContext, bestCurrentContext) and self.changeContextDecision()
            regions = regionsChangedContext if changeContext else regionsCurrentContext

            # probs = np.array([np.abs(region.evaluation) for region in regions])

            # Sort region by score
            # ids = np.argsort(-probs)[:self.options['numberRegionSelection']]
            # probs = probs[ids]
            # regions = np.array(regions)[ids]

            # if len(probs) == 0 or probs[0] == 0.:
            #     return None, False

            # Pick a region with a uniform distribution with probabilities 'probs'
            return random.choices(regions, weights=[abs(region.evaluation) for region in regions])[0], changeContext
        else:
            changeContext = self.changeContextDecision()
            regions = regionsChangedContext if changeContext else regionsCurrentContext

            if not regions:
                return None, False

            return random.choice(regions), changeContext
    
    # def _regionSamplingProbs(self, samples):
    #     # Compute score for each region
    #     probs = np.array([np.abs(sample.region.evaluation) for sample in samples])

    #     # Sort region by score
    #     ids = np.argsort(-probs)[:self.options['numberRegionSelection']]
    #     probs = probs[ids]
    #     samples = np.array(samples)[ids]

    #     if len(probs) == 0 or probs[0] == 0.:
    #         return None

    #     # Pick a region with a uniform distribution with probabilities 'probs'
    #     return uniformRowSampling(samples, probs)

    # def sampleRandomRegion2(self):
    #     """Choose one of the best regions and its most adapted strategy."""
    #     irs = []
    #     probs = []
    #     #interests = []
    #     for model in self.dataset.models:
    #         for strategy, map in model.interestMaps.items():
    #             for region in map.regions:
    #                 if region.evaluation != 0.0:
    #                     irs.append([model, strategy, region, region.evaluation])
    #                     #interests.append(region.interest)
    #                     probs.append(math.fabs(region.evaluation))  # Progresses are positive or negative
    #                     #prob_sum += math.fabs(i.interest[strat])
    #     #ids = range(len(probs))
    #     #ids.sort(key= lambda i: probs[i])
    #     probs = np.array(probs)
    #     #n = max(0, len(probs)-self.options['nb_candidates'])
    #     #ids = ids[n:len(ids)]
    #     #probs = probs[ids]
    #     #prob_sum = np.sum(probs)
    #     #probs /= prob_sum
    #     ids = list(range(len(probs)))
    #     ids.sort(key=lambda i: -probs[i])
    #     ids = ids[0:min(len(ids), self.options['numberRegionSelection'])]
    #     if len(probs) > 0 and probs[ids[0]] > 0.0:
    #         # At least one region is non-empty & could be chosen
    #         #for p in probs:
    #         #    p /= prob_sum
    #         k = ids[uniformSampling(probs)]
    #
    #         #print probs[n], ir[n]
    #         # Return result as ([model, strat, interest_region], max_interest)
    #         return tuple(irs[k])
    #     else:
    #         # All regions are empty or could not be chosen
    #         return None, None, None, 0.0

    # def __deepcopy__(self, a):
    #     newone = type(self).__new__(type(self))
    #     # newone.__dict__.update(self.__dict__)
    #     # newone.parent = None
    #     # newone.modules = []
    #     # newone.dataset = None
    #     # newone.__dict__ = copy.deepcopy(newone.__dict__)
    #     return newone

    # # deprecated
    # def plot_list_reg_v2(self, space, strat, ax, options):
    #     """Plot the regions for the given model space and strategy (only for 1D or 2D model spaces)."""
    #     it = self.space_to_interesttree(space)
    #     norm = 0.0
    #     for r in self.maps[it].regions:
    #         for i in r.interest:
    #             norm = max(-i, norm)
    #     for r in self.maps[it].regions:
    #         if not r.split:
    #             r.plot_v2(strat, norm, ax, options)


class InterestRegion(SpaceRegion):
    """Implements an interest region."""
    BASE_INTEREST = 1.
    SAMPLE_AROUND_CONTEXT = 0.3

    def __init__(self, space, options, bounds=None, parent=None, manager=None, model=None, strategy=None, contextSpace=None, regions=None, tag=None):
        super().__init__(space, options, bounds=bounds, parent=parent, manager=manager, tag='interest',
                         contextSpace=contextSpace, regions=regions)
        self._model = model
        self._strategy = strategy

    @property
    def model(self):
        return self.root()._model
    
    @property
    def strategy(self):
        return self.root()._strategy

    def updatePoint(self, point):
        if point.needUpdate():
            error, _ = self.model.forwardError(point.id)
            point.addValue(error)

    # Sample
    def checkGoalContext(self, goal, goalContext, context, changeContext):
        if not changeContext or self.number == 0:
            goalContext = None
        return goal.setRelative(True), self.dataset.controlContext(goalContext, context)

    def sampleRandomPoint(self, context=None, changeContext=True):
        """Choose a random goal inside the region."""
        goal = self.createRandomPoint()
        goalContext = self.getGoalContext(self.findRandom())

        return self.checkGoalContext(goal, goalContext, context, changeContext)
    
    def sampleGoodPoint(self, context=None, changeContext=True):
        """Find point with the highest progress inside region."""
        if not self.points:
            print("Error: bestPoint should not be used for empty region.")

        point, changeContext = self.findBest(context, changeContext)
        aroundContext = random.uniform(0, 1) < self.SAMPLE_AROUND_CONTEXT
        goal, goalContext = self.pointContextAround(point, aroundContext=aroundContext)

        return self.checkGoalContext(goal, goalContext, context, changeContext)

    def sampleBestPoint(self, context=None, changeContext=True):
        """Find point with the highest progress inside region."""
        if not self.points:
            print("Error: bestPoint should not be used for empty region.")

        point, changeContext = self.findBest(context, changeContext)
        goalContext = self.getGoalContext(point)

        return self.checkGoalContext(point.position, goalContext, context, changeContext)

    def pointContextAround(self, point, around=0.03, context=None, aroundContext=True):
        """Sample a goal around the given point inside region and a ball."""

        goal = self.pointAround(point.position, around)
        goalContext = self.getGoalContext(point)
        if aroundContext:
            goalContext = self.pointAround(goalContext, around)

        return goal, goalContext
    
    def pointAround(self, point, around=0.03):
        vect = np.random.uniform(-1.0, 1.0, point.space.dim)
        vect *= random.uniform(0., around * self.maxDistance()) / np.sqrt(np.sum(vect**2))

        data = point.npPlain() + vect

        return point.space.asTemplate(data)
    
    # def separateContext(self, point, controlContext=True):
    #     if controlContext:
    #         return point.projection(self.targetSpace).setRelative(True), point.projection(self.contextSpace, kindSensitive=True).setRelative(False)
    #     return point.projection(self.targetSpace).setRelative(True), None
