def random_search():
    args = parse_args()
    rng = np.random.RandomState(seed=args.hyp_seed)

    args.actor_alpha = rng.uniform(low=1e-4, high=1e-3)
    args.critic_alpha = rng.uniform(low=1e-4, high=1e-2)
    args.beta1 = rng.choice([0., 0.9])
    args.betas = [args.beta1, 0.999]
    args.entropy_coeff = rng.uniform(low=3e-4, high=1e-1)
    args.gamma = rng.choice([0.9, 0.95, 0.995, 1])
    args.l2_reg = 1e-4
    nn_units_actor = pow(2, rng.randint(low=5, high=9))
    nn_units_critic = pow(2, rng.randint(low=7, high=10))

    args.actor_nn_params = {
        'mlp': {
            'hidden_sizes': [nn_units_actor, nn_units_actor],
            'activation': "relu",
        }
    }
    args.critic_nn_params = {
        'mlp': {
            'hidden_sizes': [nn_units_critic, nn_units_critic],
            'activation': "relu",
        }
    }

    if args.device == 'cpu':
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create subfolder
    args.work_dir = os.path.join(args.work_dir, str(args.hyp_seed))

    # Torch Shenanigans fix
    set_one_thread()
    runner = IncrementalExpt(args)
    runner.run()