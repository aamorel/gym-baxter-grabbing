from gym.envs.registration import register

register(id='baxter_grabbing-v0',
         entry_point='gym_baxter_grabbing.envs:Baxter_grabbingEnv',)

register(id='baxter_grabbing-v1',
         entry_point='gym_baxter_grabbing.envs:Baxter_grabbingEnvOrientation',)

register(id='baxter_grabbing-v2',
         entry_point='gym_baxter_grabbing.envs:Baxter_grabbingEnvOrientationHer',)

register(id='pepper_grasping-v0',
         entry_point='gym_baxter_grabbing.envs:PepperGrasping',)
register(id='kuka_grasping-v0',
         entry_point='gym_baxter_grabbing.envs:KukaGrasping',)