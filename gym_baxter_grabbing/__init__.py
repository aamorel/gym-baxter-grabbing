from gym.envs.registration import register

register(id='baxter_grasping-v0',
         entry_point='gym_baxter_grabbing.envs:BaxterGrasping',)

register(id='pepper_grasping-v0',
         entry_point='gym_baxter_grabbing.envs:PepperGrasping',)

register(id='kuka_grasping-v0',
         entry_point='gym_baxter_grabbing.envs:KukaGrasping',)
register(id='crustcrawler-v0', entry_point='gym_baxter_grabbing.envs:CrustCrawler', max_episode_steps=200)
