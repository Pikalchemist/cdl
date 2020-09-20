import sys
from exlab.interface.serializer import Serializable


class Mod(Serializable):
    """Implements the SGIM sampling mods."""
    name = "Random"

    def __init__(self, prob):
        """
        sample func: function called when sampling with this mod
        prob float: probability of this mod to occur
        """
        self.prob = prob
        self.agent = None

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['prob'], exportPathType=True)
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options=None, obj=None):
    #     cls_ = getattr(sys.modules[__name__], dict_['type'])
    #     obj = obj if obj else cls_(dict_.get('prob'))
    #     return obj

    def sample(self, context=None):
        raise Exception('Unimplemented')

    def __repr__(self):
        return f"Mod {self.__class__.name}"


class RandomMod(Mod):
    name = "Random"

    def sample(self, context=None):
        return self.agent.interestModel.sampleRandomPoint(self.agent.trainStrategies)


class GoodRegionMod(Mod):
    """Mod corresponding to SGIM Mod 1."""
    name = "Good Region"

    def sample(self, context=None):
        return self.agent.interestModel.sampleGoodPoint(self.agent.trainStrategies, context=context)


class GoodPointMod(Mod):
    """Mod corresponding to SGIM Mod 2."""
    name = "Good Point"

    def sample(self, context=None):
        return self.agent.interestModel.sampleBestPoint(self.agent.trainStrategies, context=context)


class ActionMod(Mod):
    """Mod corresponding to random actions."""
    name = "Action Mod"

    def sample(self, context=None):
        return self.agent.interestModel.sampleRandomAction(self.agent.trainStrategies)
