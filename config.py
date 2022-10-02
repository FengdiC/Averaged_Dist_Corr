from Agents import batch_ac,weighted_batch_ac,ppo,weighted_ppo

agents_dict = {"batch_ac":batch_ac.BatchActorCritic,
               "weighted_batch_ac":weighted_batch_ac.WeightedBatchActorCritic,
               'ppo':ppo.PPO,
               'weighted_ppo':weighted_ppo.WeightedPPO,
               "batch_ac_shared_gc": weighted_batch_ac.SharedWeightedCriticBatchAC,
               'weighted_shared_batch_ac':weighted_batch_ac.SharedWeightSeparateUpdateBatchAC,
               'ppo_shared_gc':weighted_ppo.SharedWeightedPPO,
               'weighted_shared_ppo':weighted_ppo.WeightSeparateUpdatePPO}