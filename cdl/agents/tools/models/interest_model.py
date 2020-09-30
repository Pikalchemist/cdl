from dino.agents.tools.models.regression import RegressionModel


class InterestModel(RegressionModel):
    def __init__(self, dataset, actionSpace, outcomeSpace, contextSpace=[], restrictionIds=None, model=None,
                 register=True):
        super().__init__(dataset, actionSpace, outcomeSpace, contextSpace,
                         restrictionIds, model, register)

        self.interestMaps = {}
        # if model:
        #     for name, s_previous, s_current in [('a', model.actionSpace, self.actionSpace), ('o', model.outcomeSpace, self.outcomeSpace), ('c', model.contextSpace, self.contextSpace)]:
        #         if model.actionSpace != self.actionSpace:
        #             self.spacesHistory.append(self.dataset.getIteration(), EventKind.DELETE, [("{}-{}".format(s_previous.id),)])
        #             self.spacesHistory.append(self.dataset.getIteration(), EventKind.ADD, [("{}-{}".format(s_current.id), concat(s_current.serialize(), {'kind': name}))])
        # else:
        #     for name, s in [('a', self.actionSpace), ('o', self.outcomeSpace), ('c', self.contextSpace)]:
        #         self.spacesHistory.append(self.dataset.getIteration(), DataEventKind.ADD, [("{}-{}".format(name, s.id), concat(s.serialize(), {'kind': name}))])

    def interestMap(self, index=0):
        return list(self.interestMaps.items())[index][1]
    
    def continueFrom(self, previousModel):
        super().continueFrom(previousModel)

        # self.interestMaps = previousModel.interestMaps
        # previousModel.spacesHistory.extend(self.spacesHistory)
        # self.spacesHistory = previousModel.spacesHistory
