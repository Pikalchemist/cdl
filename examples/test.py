import io
import pstats
import cProfile
from pstats import SortKey

from dinos.environments.playground import PlaygroundEnvironment
from cdl.agents.learners.curiosity.saggriac import SAGGLearner
from cdl.agents.tools.models.interest_model import InterestModel

from exlab.interface.graph import display
from dinos.utils.move import MoveConfig
from dinos.agents.tools.planners.planner import Planning


env = PlaygroundEnvironment()
learner = SAGGLearner(env.world.findHost())
learner.adaptiveModels = False

mMove = InterestModel(learner.dataset, learner.propertySpace(
    'Agent.move'), learner.propertySpace('Agent.position'), learner.propertySpace('Agent.lidar'))
mRelativeObject = InterestModel(learner.dataset, learner.propertySpace('Agent.position'), learner.propertySpace(
    '#Cylinder1.positionToAgent'), learner.propertySpace('#Cylinder1.positionToAgent'))
mObject = InterestModel(learner.dataset, learner.propertySpace('Agent.position'), learner.propertySpace(
    '#Cylinder1.position'), learner.propertySpace('#Cylinder1.positionToAgent'))
mButton = InterestModel(learner.dataset, learner.propertySpace('Agent.position'), learner.propertySpace(
    '#Button1.pressed'), learner.propertySpace('#Cylinder1.position'))

mObject.limitMoves = 0.3

for _ in range(1):
    learner.train(100)
    env.run()


pr = cProfile.Profile()
pr.enable()

for _ in range(1):
    learner.train(1000)
    env.run()

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats()
lines = s.getvalue().split('\n')
print(learner.iteration)
print(Planning.DEBUG)
print('\n'.join(lines[:100]))
# print('\n'.join([line for line in lines if 'regression' in line or 'model.py' in line or 'contextarea.py' in line]))
