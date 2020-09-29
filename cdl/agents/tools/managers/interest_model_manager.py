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
        super().__init__("InterestModelManager", learner)
        # Different keys of options
        # - numberRegionSelection: maximum number of best regions considered when choosing with intrinsic motivation
        # - around: radius of the ball used to sample around a specific point
        # - costs: list containing the cost of each available strategy for the learner
        # - ## contains other options described in InterestRegion class documentation
        self.dataset = learner.dataset
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

    # @classmethod
    # def _deserialize(cls, dict_, dataset, loadResults=True, options={}, obj=None):
    #     obj = cls(dataset, dict_.get('options', {}))
    #     return obj

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
        models = [model for model in self.dataset.models
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
            contextColumns = model.contextColumns(None, outcome, context.projection(model.contextSpace))
            # print(
            #     f'action {action} actions {actions} model {model}')
            predictedOutcome = model.forward(action, context=context, contextColumns=contextColumns)[0]
            if predictedOutcome:
                # print(f'{predictedOutcome} {outcome}')
                # print(f'predicted {predictedOutcome.plain()} and got {outcome.plain()}')
                distance = predictedOutcome.distanceTo(outcome) * 10. / outcome.space.maxDistance
                # print(f'{distance}: {model}')

                if strategy not in model.interestMaps:
                    self.createInterestMap(model, strategy)
                model.interestMaps[strategy].addPoint(
                    outcomes.extends(context), distance)
                model.pointAdded(event, distance)

        # progresses = [self.computeProgress(previousCompetence, model.goalCompetence(outcomes, context))
        #               for model, previousCompetence in zip(models, competences)]
        # for model, progress in zip(models, progresses):
        #     if strategy not in model.interestMaps:
        #         self.createInterestMap(model, strategy)
        #     model.interestMaps[strategy].addPoint(
        #         outcomes.extends(context), progress)
        #     model.pointAdded(event, progress)

    def computeProgress(self, from_, to):
        return (to - from_) * (max(to, from_) ** 2)

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
            return [m for m in self.dataset.models if strategiesAvailable.intersection(set(m.interestMaps.keys()))]
        return [m for m in self.dataset.models if m.interestMaps]

    # Samples
    def sampleRandomAction(self, strategiesAvailable=[]):
        """Sample action."""
        if not strategiesAvailable:
            models = self.modelsWithInterestMaps()
            if models:
                strategiesAvailable = set([strategy for model in models for strategy, _ in model.interestMaps.items()])

        strategy = random.choice(list(strategiesAvailable))
        return MoveConfig(strategy=strategy, sampling="random space, random action")

    def sampleRandomPoint(self, strategiesAvailable=None, context=None):
        """Sample one point at random in the regions."""
        region, controlContext = self.sampleRandomRegion(
            strategiesAvailable=strategiesAvailable, context=context)
        if region is None:
            return self.sampleRandomAction(strategiesAvailable=strategiesAvailable)

        # models = self.modelsWithInterestMaps(
        #     strategiesAvailable=strategiesAvailable)
        # if not models:
        #     return self.sampleRandomAction(strategiesAvailable=strategiesAvailable)

        # model = random.choice(models)
        # maps = model.interestMaps
        # if strategiesAvailable:
        #     maps = {strategy: mp for strategy,
        #             mp in maps.items() if strategy in strategiesAvailable}
        # strategy = random.choice(list(maps.keys()))
        # mp = maps[strategy]

        # if not mp.controllableContext(self.dataset):
        #     sampleContext = False
        # else:
        #     sampleContext = self.sampleContext()

        # goal, goalContext = mp.randomPoint(
        #     context=context, sampleContext=sampleContext)
        # goalContext = self.dataset.controlContext(goalContext, context)
    
        goal, goalContext = region.separateContext(region.randomPoint(), controlContext)
        goalContext = self.dataset.controlContext(goalContext, context)

        return MoveConfig(model=region.model, strategy=region.strategy, goal=goal, goalContext=goalContext,
                          sampling="Random region, random goal")

    def sampleGoodPoint(self, strategiesAvailable=[], context=None):
        """Sample one point at random in one of the best regions."""
        region, controlContext = self.sampleBestRegion(
            strategiesAvailable=strategiesAvailable, context=context)
        if region is None:
            return self.sampleRandomPoint(strategiesAvailable=strategiesAvailable)

        goal, goalContext = region.separateContext(region.randomPoint(), controlContext)
        goalContext = self.dataset.controlContext(goalContext, context)

        return MoveConfig(model=region.model, strategy=region.strategy, goal=goal, goalContext=goalContext,
                          sampling=f"best region in space {region.space} with an interest of {region.evaluation:.4f}, random goal")

    def sampleBestPoint(self, strategiesAvailable=[], context=None):
        """Sample one point around the best point in one of the best regions."""
        region, controlContext = self.sampleBestRegion(
            strategiesAvailable=strategiesAvailable, context=context)
        if region is None:
            return self.sampleRandomPoint(strategiesAvailable=strategiesAvailable)

        goal, goalContext = region.separateContext(region.pointAround(
            self.options['around'], context=context, controlContext=controlContext, aroundContext=False), controlContext)
        goalContext = self.dataset.controlContext(goalContext, context)

        return MoveConfig(model=region.model, strategy=region.strategy, goal=goal, goalContext=goalContext,
                          sampling=f"best region in space {region.space} with an interest of {region.evaluation:.4f}, goal around best point")

    def controlContext(self, bestControlledContext, samplesCurrentContext):
        if samplesCurrentContext > bestControlledContext * 4:
            return False
        if bestControlledContext > samplesCurrentContext * 4:
            return True
        return random.uniform(0, bestControlledContext + samplesCurrentContext) < bestControlledContext
    
    def sampleBestRegion(self, strategiesAvailable=[], context=None):
        return self.sampleRandomRegion(strategiesAvailable, context, best=True)

    def sampleRandomRegion(self, strategiesAvailable=[], context=None, best=False):
        # List all available regions
        regions = []
        for model in self.dataset.models:
            for strategy, mp in model.interestMaps.items():
                if not strategiesAvailable or strategy in strategiesAvailable:
                    for region in mp.regions:
                        if region.evaluation != 0.0:
                            regions.append(region)

        # Check if region context may be controlled
        regionsControlledContext = []
        for region in regions:
            if region.controllableContext(self.dataset):
                regionsControlledContext.append(region)
        bestControlledContext = np.max([np.abs(region.evaluation) for region in regionsControlledContext]) if regionsControlledContext else 0
            # if regionsTemp:
            #     regions = regionsTemp
            # else:
            #     sampleContext = False

        # Filter by context
        # if not sampleContext and context:
        regionsCurrentContext = []
        if context:
            for region in regions:
                if region.nearContext(context):
                    regionsCurrentContext.append(region)
        bestCurrentContext = np.max([np.abs(region.evaluation) for region in regionsCurrentContext]) if regionsCurrentContext else 0

        if not regionsControlledContext and not regionsCurrentContext:
            return None, False

        if best:
            controlContext = self.controlContext(bestControlledContext, bestCurrentContext)
            regions = regionsControlledContext if controlContext else regionsCurrentContext

            probs = np.array([np.abs(region.evaluation) for region in regions])

            # Sort region by score
            ids = np.argsort(-probs)[:self.options['numberRegionSelection']]
            probs = probs[ids]
            regions = np.array(regions)[ids]

            if len(probs) == 0 or probs[0] == 0.:
                return None, False

            # Pick a region with a uniform distribution with probabilities 'probs'
            return uniformRowSampling(regions, probs), controlContext
        else:
            controlContext = random.uniform(0, 2) < 1
            regions = regionsControlledContext if controlContext else regionsCurrentContext
            if not regions:
                return None, False
            region = random.choice(regions)
            return region, controlContext
    
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
    BASE_INTEREST = 2.

    def __init__(self, space, options, bounds=None, parent=None, manager=None, model=None, strategy=None, contextSpace=None, regions=None):
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

    def computeEvaluation(self):
        """Compute evaluation of the region for given strategy."""
        if len(self.pointValues) > self.options['window'] // 2:
            self.progresses.append(self.meanProgress(self.pointValues))
        if len(self.progresses) > self.options['window'] // 2:
            self.evaluation = np.abs((self.progresses[-1] - self.progresses[-self.options['window'] // 2]) / self.options['cost'])
        elif self.parent:
            self.evaluation = 0.
        else:
            self.evaluation = self.BASE_INTEREST

    def meanProgress(self, pointValues):
        """Compute mean progress according to the evaluation window."""
        pointValues = np.sort(pointValues[-self.options['window']//2:])
        # if len(pointValues) > 5:
        #     pointValues = pointValues[1:-1]
        return np.mean(pointValues)

    # Sample
    def bestPoint(self, context=None, controlContext=True):
        """Find point with the highest progress inside region."""
        if not self.pointValues:
            print("Error: bestPoint should not be used for empty region.")

        if context and not controlContext:
            ids, dists = self.nearestContext(context)
            indices, _ = mixedSort(dists, -np.abs(self.pointValues)[ids])
        else:
            indices = (-np.abs(self.pointValues)).argsort()

        return self.space.asTemplate(self.points[indices[0]])

    def pointAround(self, around=0.05, point=None, context=None, controlContext=True, aroundContext=False):
        """Sample a goal around the given point inside region and a ball."""
        point = parameter(self.bestPoint(
            context=context, controlContext=controlContext))
        point = point.npPlain()

        # TODO: or maxDistance of the region?
        around = around * self.maxDistance()

        # Compute a random normalized vector indicating exploration direction
        vect = np.random.uniform(-1.0, 1.0, len(point))
        vect *= random.uniform(0., around) / np.sqrt(np.sum(vect**2))
        if aroundContext:
            point += vect
        else:
            cols = self.space.columnsFor(self.targetSpace)
            point[cols] += vect[cols]

        # Check if the point is still in the evaluation region
        for i in range(len(point)):
            point[i] = max(min(point[i], self.bounds[i][1]), self.bounds[i][0])

        return self.space.asTemplate(point)

    def randomPoint(self):
        """Choose a random goal inside the region."""
        d = Data(self.space,
                 [random.uniform(bmin, bmax) for bmin, bmax in self.finiteBounds()])

        return d
    
    def separateContext(self, point, controlContext=True):
        if controlContext:
            return point.projection(self.targetSpace), point.projection(self.contextSpace)
        return point.projection(self.targetSpace), None
