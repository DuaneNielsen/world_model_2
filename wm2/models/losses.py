from torch.distributions import Normal

from worldmodel import args


def mse_loss(trajectories, predicted_state):
    loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
        args.device)
    return loss.mean()


def log_prob_loss_simple(trajectories, predicted_state):
    return - predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()


def log_prob_loss_entropy(trajectories, predicted_state):
    prior = Normal(predicted_state.loc[0:-1], predicted_state.scale[0:-1])
    posterior = Normal(predicted_state.loc[1:], predicted_state.scale[1:])
    # div = kl_divergence(prior, posterior).mean()
    log_p = predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()
    entropy = posterior.entropy().mean()
    return - log_p - entropy