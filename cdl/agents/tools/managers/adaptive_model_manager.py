import random
import numpy as np
from enum import Enum, auto

from exlab.modular.module import Module

from dino.data.space import SpaceKind


class ModelMutation(Enum):
    ADD_SPACE = auto()
    DELETE_SPACE = auto()
    CHANGE_SPACE = auto()
    CONTEXT_SPACE = auto()
    ACTION_SPACE = auto()


class AdaptiveModelManager(Module):
    MODEL_LOW_THRESHOLD = 0.4
    MODEL_LOW_PROGRESSION = 0.2
    MODEL_LOW_PERIOD = 50

    MODEL_SIMILAR_DURATION = 50
    MODEL_SIMILAR_THRESHOLD = 0.05

    EVALUATION_WINDOW = 6

    def __init__(self, learner):
        super().__init__('AdaptiveModelManager')
        self.logger.tag = 'adaptation'
        self.learner = learner
        self.dataset = self.learner.dataset

        self.creationCriterium = 0.3
        # self.deletionCriterium = 0.4
        self.spaceAdditionCriterium = 0.1
        self.spaceDeletionCriterium = 0.01

        self.iterationStart = 10
        self.stackEvents = 50
        self.eventStacked = []

    def addEvent(self, event):
        if self.learner.iteration < self.iterationStart:
            return

        self.eventStacked.append(event)
        if len(self.eventStacked) >= self.stackEvents:
            print('******************************')
            #events = [event for list_event in events for event in list_event]
            #actionSpaces = set([aspace for event in events for aspace in event.actions.get_space().iterate()])
            self.analyseEvents(self.eventStacked)

            self.evaluateModels(self.eventStacked)

            self.createModels(self.eventStacked)
            self.updateModels(self.eventStacked)
            # self.mergeModels(self.eventStacked)

            self.eventStacked = []

            #print(self.model_history)

            #import sys
            #sys.exit()

    def analyseEvents(self, events):
        # retrieve all spaces
        self.actionSpaces = set()
        self.outcomeSpaces = set()
        for event in events:
            self.actionSpaces.update(set(event.actions.space.iterate()))
            self.outcomeSpaces.update(set(event.outcomes.space.iterate()))

        self.logger.debug2(f'All spaces considered: {self.outcomeSpaces}')

        # filter by non-existing models
        nonAlreadyExistingSpaces = [space for space in self.outcomeSpaces]
        #if not self.dataset.findModelByOutcomeSpace(space)]

        # self.logger.debug2(f'Filtered by non existing models: {nonAlreadyExistingSpaces}')

        # filter by variation
        variatingSpaces = []
        for space in nonAlreadyExistingSpaces:
            data = [(event, event.outcomes.projection(space)) for event in events
                    if event.outcomes.projection(space).norm() > 1.]
            if data:
                variatingSpaces.append(space)  # (space, data))

        self.logger.debug2(f'Filtered by variating only spaces: {[space for space in variatingSpaces]}')

        self.actionSpaces.update(
            self.dataset.controllableSpaces(self.outcomeSpaces, performant=True))
        self.variatingOutcomeSpaces = variatingSpaces
        self.learnableOutcomeSpaces = [space for space in variatingSpaces if space.learnable]

        self.variatingContextSpaces = [space.convertTo(kind=SpaceKind.PRE) for space in variatingSpaces]

    def createModels(self, events):
        self.logger.debug2(f'Trying to add new models at iteration {self.learner.iteration} on a set of {len(events)} events')
        # ids = [event.id for event in events]

        if not self.learnableOutcomeSpaces or not self.actionSpaces:
            return

        # filter by competence
        candidateModels = {}
        testedModels = []
        for _ in range(5):
            outcomeSpace = random.choice(self.learnableOutcomeSpaces)
            # print('--------')
            # print('Considering space {}'.format(outcomeSpace))

            possibleSingleActionSpaces = set([space
                                              for space in self.actionSpaces if not space.intersects(outcomeSpace)])
            possibleSingleContextSpaces = set([self.dataset.convertSpace(space, SpaceKind.PRE)
                                               for space in self.variatingOutcomeSpaces if not space.intersects(outcomeSpace)])
            if not possibleSingleActionSpaces:
                continue

            # print('possibleSingleActionSpaces {}'.format(possibleSingleActionSpaces))
            # print('possibleSingleContextSpaces {}'.format(possibleSingleContextSpaces))

            model = None
            for j in range(5):
                if model is None or not model.contextSpace:
                    # Random selection
                    actionSpace = random.choice(list(possibleSingleActionSpaces))
                    if j == 0 or not possibleSingleContextSpaces:  # test with no context space
                        contextSpace = []
                    else:
                        contextSpace = [random.choice(
                            list(possibleSingleContextSpaces))]
                else:
                    # Test previous model without context
                    contextSpace = []

                #[[s] for s in self.learner.dataset.spaces if not s.intersects(space) and s.kind == SpaceKind.PRE] + [[]]
                model = self.dataset.modelClass(self.dataset,
                                                actionSpace,
                                                outcomeSpace,
                                                contextSpace,
                                                register=False)  # restrictionIds=ids,
                if model.contextSpacialization:
                    model.contextSpacialization[0].allTrue()

                # Checks that the model does not already exist neither has already been tested
                if [True for m in testedModels if m.matches(model)]:
                    continue
                testedModels.append(model)
                if [True for m in self.dataset.models if m.matches(model)]:
                    continue
            
                if self.dataset.isGraphCyclic(self.dataset.dependencyGraph(list(self.dataset.enabledModels()) + [model])):
                    continue

                # print(m)
                competence = model.competence()
                if competence >= self.creationCriterium:
                    competence = model.competence(precise=True)
                # favors no context models
                score = competence + 0.02 * (len(contextSpace) == 0)
        
                if [True for m in self.dataset.models if m.matches(model, ignoreContext=True) and competence < m.competence(precise=True) + 0.05]:
                    # print(f'//// failed for {model}')
                    continue
                # Checks that the new model does not introduce a cycle in the model dependency graph
                

                

                # Checks if an other model with different context space exists and performs better
                # if max([m.competence() for m in self.dataset.models if m.matches(model, ignoreContext=True)] + [0]) >= competence:
                #     continue

                # print(f'**** Score {score} for model {model}')
                if score >= self.creationCriterium:
                    if outcomeSpace not in candidateModels:
                        candidateModels[outcomeSpace] = []
                    candidateModels[outcomeSpace].append((model, score))
                    if len(contextSpace) > 0:
                        j -= 1

        if not candidateModels:
            return
        for outcomeSpace, models in candidateModels.items():
            ms = np.array(models)
            candidateModels[outcomeSpace] = ms[ms[:, 1].argsort()][::-1, :]

            # print('Best candidates for {}: {}'.format(outcomeSpace, candidateModels[outcomeSpace]))
            # print('====================')

        # best candidates
        for outcomeSpace, models in candidateModels.items():
            model, score = models[0]
            if not self.dataset.isGraphCyclic(self.dataset.dependencyGraph(list(self.dataset.enabledModels()) + [model])):
                # self.log_model_change("Adding model : {}".format(model))
                if model.contextSpacialization:
                    model.contextSpacialization[0]._resetAreas()
                self.logger.info(f'Adding {model} (with score {score})')
                self.learner.dataset.registerModel(model)

    def evaluateModels(self, events):
        for model in self.dataset.enabledModels():
            model.competence(precise=True)
        for model in list(self.dataset.models):
            self.evaluateModel(model)

    def evaluateModel(self, model, precise=True):
        if not model.enabled:
            if not random.uniform(0, 1) < 0.2:
                return
            evaluation = model.competence()
            if not evaluation > self.MODEL_LOW_THRESHOLD:
                return

        if model.enabled:
            competence = model.lastCompetence
        else:
            competence = model.competence(precise=precise)

        # Compare to other similar models (same action and outcome spaces)
        if model.enabled:
            if model.duration > self.MODEL_SIMILAR_DURATION:
                similarModels = [m for m in self.dataset.enabledModels() if m.matches(
                    model, ignoreContext=True) and m != model and m.duration > self.MODEL_SIMILAR_DURATION]

                similarCompetences = [m.lastCompetence for m in similarModels]
                if similarCompetences and max(similarCompetences) - competence > self.MODEL_SIMILAR_THRESHOLD:
                    model.enabled = False

        # Competence and progress
        model.evaluations[self.learner.iteration] = competence
        if len(model.evaluations) > self.EVALUATION_WINDOW:
            oldest = min(list(model.evaluations.keys()))
            del model.evaluations[oldest]
        # while True:
        #     oldest = min(list(model.evaluations.keys()))
        #     if self.learner.iteration - oldest > self.EVALUATION_WINDOW:
        #         del model.evaluations[oldest]
        #     else:
        #         break

        if len(model.evaluations) > 3:
            evaluations = np.array([value for key, value in sorted(model.evaluations.items())])
            maxRecentEvaluation = np.max(evaluations[-2:])
            progression = maxRecentEvaluation - np.max(evaluations[:-2])

            # print(evaluations)
            # print(maxRecentEvaluation)
            # print(progression)

            if maxRecentEvaluation < self.MODEL_LOW_THRESHOLD and progression < self.MODEL_LOW_PROGRESSION:
                # print('low!')
                # print(self.learner.iteration - model.lowCompetenceSince)
                # print(model.lowCompetenceSince >= 0 and self.learner.iteration -
                #       model.lowCompetenceSince > self.MODEL_LOW_PERIOD)
                if model.lowCompetenceSince >= 0 and self.learner.iteration - model.lowCompetenceSince > self.MODEL_LOW_PERIOD:
                    model.enabled = False
                    self.logger.info(f'Disabling {model}')
                if model.lowCompetenceSince == -1:
                    model.lowCompetenceSince = self.learner.iteration
            else:
                model.lowCompetenceSince = -1
                if not model.enabled:
                    if not self.dataset.isGraphCyclic(self.dataset.dependencyGraph(list(self.dataset.enabledModels()) + [model])):
                        model.enabled = True
                        self.logger.info(f'Re-enabling {model}')

            # if model.lowCompetenceSince >= 0 and self.learner.iteration - model.lowCompetenceSince > self.MODEL_LOW_PERIOD * 6:
            #     self.logger.info(
            #         f'Deleting {model} (because of a score of {lastEvaluation})')
            #     self.learner.dataset.unregisterModel(model)

    def updateModels(self, events):
        for model in self.dataset.enabledModels():
            currentCompetence = model.competence()
            currentPreciseCompetence = None

            # Try mutations on the model
            number = random.randint(2, 6)
            models = set(self.dataset.enabledModels())
            models.remove(model)
            for _ in range(number):
                newModel, deletion = self.mutateModel(model)
                # print(f'@@@@ Attempt {model} -> {newModel}')

                if not newModel:
                    continue
                if self.dataset.isGraphCyclic(self.dataset.dependencyGraph(list(models | set([newModel])))):
                    continue

                criterium = self.spaceDeletionCriterium if deletion else self.spaceAdditionCriterium
                if newModel.competence() > currentCompetence + criterium:
                    if currentPreciseCompetence is None:
                        currentPreciseCompetence = model.competence(precise=True)
                    preciseCompetence = newModel.competence(precise=True)
                    # print(f'@@@@ => {preciseCompetence} {currentPreciseCompetence}')
                    if preciseCompetence > currentPreciseCompetence + self.spaceAdditionCriterium:
                        self.dataset.replaceModel(model, newModel)
                        self.logger.info(f'Modified {model} into {newModel} (for a gain of {preciseCompetence - currentPreciseCompetence})')
                        break  # only one modification per model per iteration
    
    def mutateModel(self, model):
        availableMutations = [ModelMutation.ADD_SPACE]
        if model.contextSpace:
            availableMutations.append(ModelMutation.CHANGE_SPACE)

        mutation = random.choice(availableMutations)
        newModel = None
        deletion = False

        actionSpaceSet = set(model.actionSpace.iterate())
        outcomeSpaceSet = set(model.outcomeSpace.iterate())
        contextSpaceSet = set(model.contextSpace.iterate())

        if mutation == ModelMutation.ADD_SPACE or mutation == ModelMutation.CHANGE_SPACE:
            kind = random.choice((ModelMutation.CONTEXT_SPACE,))

            if kind == ModelMutation.CONTEXT_SPACE:
                workingSpace = model.contextSpace
                workingSet = contextSpaceSet
                availableSpaces = list(self.variatingContextSpaces)

            availableSpaces = [
                space for space in availableSpaces if not space.intersects(workingSpace)]
            if not availableSpaces and mutation == ModelMutation.ADD_SPACE:
                return None, deletion
            if not workingSet and (mutation == ModelMutation.DELETE_SPACE or mutation == ModelMutation.CHANGE_SPACE):
                return None, deletion

            if mutation == ModelMutation.DELETE_SPACE or mutation == ModelMutation.CHANGE_SPACE:
                for _ in range(1):
                    space = random.choice(list(workingSet))
                    workingSet.remove(space)
                deletion = True
            if mutation == ModelMutation.ADD_SPACE or mutation == ModelMutation.CHANGE_SPACE:
                for _ in range(1):
                    space = random.choice(availableSpaces)
                    workingSet.add(space)
                    availableSpaces.remove(space)

            if kind == ModelMutation.CONTEXT_SPACE:
                contextSpaceSet = workingSet

        newModel = self.dataset.modelClass(self.dataset,
                                           list(actionSpaceSet),
                                           list(outcomeSpaceSet),
                                           list(contextSpaceSet),
                                           register=False)
        return newModel, deletion

    def mergeModels(self, events):
        for model in list(self.dataset.models):
            competence = model.competence()
            preciseCompetence = None
            for other in list(self.dataset.models):
                if model != other:
                    merge = False
                    otherCompetence = other.competence()
                    if model.matches(other):
                        merge = True
                    elif model.matches(other, ignoreContext=True) and otherCompetence > competence:
                        if preciseCompetence is None:
                            preciseCompetence = model.competence(precise=True)
                        if other.competence(precise=True) > preciseCompetence:
                            merge = True
                    if merge:
                        self.logger.info(f'Merging {model} (comp {competence} - deleted) with {other} (comp {otherCompetence})')
                        self.learner.dataset.unregisterModel(model)
                        break

    # def updateModels__OLD(self, events):
    #     ids = [event.id for event in events]
    #     models = list(self.models)
    #     for model in self.models:
    #         data = [event.outcomes.get_projection(
    #             model.outcomeSpace) for event in events]

    #         # Trying to add a new space
    #         sd = model.competence(data)
    #         '''print(str(model) + " has " + str(sd))
    #         print(data)
    #         print([event.actions.get()[0].parts[0].type for event in events])
    #         print(str(model.outcomeSpace.getData()[-25:, 0]))
    #         print(str(model.actionSpace.getData()[-25:, 0]))
    #         print(str(self.actionSpaces[0][0].getData()[-25:, 0]))'''

    #         actionSpaces = set(
    #             [aspace for event in events for aspace in event.actions.get_space().iterate()])
    #         for aspace in model.actionSpace.iterate():
    #             if aspace in actionSpaces:
    #                 actionSpaces.remove(aspace)
    #         if actionSpaces:
    #             actionSpacesSelection = random.choice(list(actionSpaces))
    #             actionSpaces_used = []
    #             actionSpaces_removed = []
    #             for aspace in model.actionSpace.iterate():
    #                 if random.uniform(0, 1) < 0.7:
    #                     actionSpaces_used.append(aspace)
    #                 else:
    #                     actionSpaces_removed.append(aspace)

    #             newModel = self.learner.dataset.modelClass(self.learner.dataset, actionSpaces_used + [
    #                                                      actionSpacesSelection], model.outcomeSpace.iterate(), restrictionIds=ids, model=model, register=False)
    #             nsd = newModel.competence(data)
    #             # self.log_model_change("Trying to add {} - local, sd = {}".format(actionSpacesSelection, nsd))
    #             if nsd <= sd * 0.95:
    #                 newModel = self.learner.dataset.modelClass(self.learner.dataset, newModel.actionSpace.iterate(
    #                 ), newModel.outcomeSpace.iterate(), model=model, register=False)
    #                 nsd = newModel.competence(data)
    #                 # self.log_model_change("Trying to add {} - local, sd = {}".format(actionSpacesSelection, nsd))
    #                 if nsd <= sd * 0.95:
    #                     # self.log_model_change("/!\ Modification approved {}".format(newModel.actionSpace))
    #                     # newModel.spacesHistory.append(self.getIteration(), EventKind.ADD, [("a-{}".format(actionSpacesSelection.id), concat(actionSpacesSelection.serialize(), {'kind': 'a'}))])
    #                     # if actionSpaces_removed:
    #                     # newModel.spacesHistory.append(self.getIteration(), EventKind.ADD, [("a-{}".format(space.id),) for space in actionSpaces_removed])
    #                     self.learner.dataset.replaceModel(model, newModel)
    #                     print(self.models)
    #                     sd = nsd
    #                     model = newModel

    #         # Trying to remove a space
    #         actionSpaces = model.actionSpace.iterate()
    #         if len(actionSpaces) > 1:
    #             actionSpacesSelection = random.choice(actionSpaces)
    #             actionSpaces.remove(actionSpacesSelection)

    #             newModel = self.learner.dataset.modelClass(self.learner.dataset, actionSpaces, model.outcomeSpace.iterate(
    #             ), restrictionIds=ids, model=model, register=False)
    #             nsd = newModel.competence(data)
    #             # self.log_model_change("Trying to remove {} - local, sd = {}".format(actionSpacesSelection, nsd))
    #             if nsd <= sd * 1.05:
    #                 newModel = self.learner.dataset.modelClass(self.learner.dataset, newModel.actionSpace.iterate(
    #                 ), newModel.outcomeSpace.iterate(), model=model, register=False)
    #                 nsd = newModel.competence(data)
    #                 # self.log_model_change("Trying to remove {} - local, sd = {}".format(actionSpacesSelection, nsd))
    #                 if nsd <= sd * 1.01:
    #                     # self.log_model_change("/!\ Modification approved {}".format(newModel.actionSpace))
    #                     # newModel.spacesHistory.append(self.getIteration(), EventKind.DELETE, [("a-{}".format(actionSpacesSelection.id),)])
    #                     self.learner.dataset.replaceModel(model, newModel)
    #                     sd = nsd
    #                     model = newModel

    # # def log_model_change(self, text):
    # #     self.model_history.append({'str': text})
