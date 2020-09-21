from .interest_learner import InterestLearner


class SAGGLearner(InterestLearner):
    def __init__(self, environment, mods=[], dataset=None, options={}, performer=None, planner=None):
        from dino.agents.tools.strategies.random import RandomStrategy
        from dino.agents.tools.strategies.autonomous import AutonomousStrategy
        from cdl.agents.tools.mods import RandomMod, GoodRegionMod, GoodPointMod
        mods = [RandomMod(0.2), GoodRegionMod(0.6), GoodPointMod(0.2)] + mods
        super().__init__(environment, mods=mods, dataset=dataset, options=options, performer=performer,
                         planner=planner)
        # self.trainStrategies.add(RandomStrategy(self))
        self.trainStrategies.add(AutonomousStrategy(self))
        self.testStrategies.add(AutonomousStrategy(self))
        # self.trainStrategies.append(AutonomousExploration(self))
