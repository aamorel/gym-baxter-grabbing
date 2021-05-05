from gym.envs.registration import register

register(id='baxter_grasping-v0', entry_point='gym_baxter_grabbing.envs:BaxterGrasping', max_episode_steps=5000)

register(id='pepper_grasping-v0', entry_point='gym_baxter_grabbing.envs:PepperGrasping',max_episode_steps=2000)

register(id='kuka_grasping-v0', entry_point='gym_baxter_grabbing.envs:KukaGrasping',max_episode_steps=2500)

register(id='crustcrawler-v0', entry_point='gym_baxter_grabbing.envs:CrustCrawler', max_episode_steps=2500)
