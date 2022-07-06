from Agents import batch_ac,weighted_batch_ac,ppo,weighted_ppo

agents_dict = {"batch_ac":batch_ac.BatchActorCritic,
               "weighted_batch_ac":weighted_batch_ac.WeightedBatchActorCritic,
               'ppo':ppo.PPO,
               'weighted_ppo':weighted_ppo.WeightedPPO}