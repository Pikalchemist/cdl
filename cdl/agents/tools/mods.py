import sys
from exlab.interface.serializer import Serializable


class Mod(Serializable):
    """Implements the SGIM sampling mods."""
    name = "Random"

    def __init__(self, probability, changeContextProbability=None):
        """
        sample func: function called when sampling with this mod
        prob float: probability of this mod to occur
        """
        self.agent = None
        self.probability = probability
        self.changeContextProbability = changeContextProbability

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['probability'])
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options=None, obj=None):
    #     cls_ = getattr(sys.modules[__name__], dict_['type'])
    #     obj = obj if obj else cls_(dict_.get('prob'))
    #     return obj

    def sampleMethod(self):
        raise Exception('Unimplemented')

    def sample(self, context=None):
        return self.sampleMethod()(self.agent.trainStrategies, context=context, changeContextProbability=self.changeContextProbability)

    def __repr__(self):
        return f"Mod {self.__class__.name}"


class RandomGoalMod(Mod):
    name = "Random Goal Point"

    def sampleMethod(self):
        return self.agent.interestModel.sampleRandomGoal


class GoodGoalMod(Mod):
    """Mod corresponding to SGIM Mod 1."""
    name = "Good Goal Point"

    def sampleMethod(self):
        return self.agent.interestModel.sampleGoodGoal


class BestGoalMod(Mod):
    """Mod corresponding to SGIM Mod 2."""
    name = "Best Goal Point"

    def sampleMethod(self):
        return self.agent.interestModel.sampleBestGoal


class RandomActionMod(Mod):
    """Mod corresponding to random actions."""
    name = "Random Action Point"

    def sampleMethod(self):
        return self.agent.interestModel.sampleRandomAction
