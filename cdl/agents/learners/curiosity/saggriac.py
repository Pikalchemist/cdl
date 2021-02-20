from .interest_learner import InterestLearner


class SAGGLearner(InterestLearner):
    def __init__(self, host, mods=[], dataset=None, options={}, performer=None, planner=None):
        from dino.agents.tools.strategies.random import RandomStrategy
        from dino.agents.tools.strategies.autonomous import AutonomousStrategy
        from cdl.agents.tools.mods import ActionMod, RandomMod, GoodRegionMod, GoodPointMod
        mods = [ActionMod(0.1), RandomMod(0.15), GoodRegionMod(0.55), GoodPointMod(0.2)] + mods
        super().__init__(host, mods=mods, dataset=dataset, options=options, performer=performer,
                         planner=planner)
        # self.trainStrategies.add(RandomStrategy(self))
        self.trainStrategies.add(AutonomousStrategy(self))
        self.testStrategies.add(AutonomousStrategy(self))
        # self.trainStrategies.append(AutonomousExploration(self))
