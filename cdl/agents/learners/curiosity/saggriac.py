from .interest_learner import InterestLearner


class SAGGLearner(InterestLearner):
    def __init__(self, environment, mods=[], dataset=None, options={}, performer=None, planner=None):
        from dino.agents.tools.strategies.random import RandomStrategy
        # from dino.agents.tools.strategies.autonomous_exploration import AutonomousExploration
        from cdl.agents.tools.mods import RandomMod, GoodRegionMod, GoodPointMod
        mods = [RandomMod(0.2), GoodRegionMod(0.6), GoodPointMod(0.2)] + mods
        super().__init__(environment, mods=mods, dataset=dataset, options=options, performer=performer,
                         planner=planner)
        self.trainStrategies.append(RandomStrategy(self))
        # self.trainStrategies.append(AutonomousExploration(self))
