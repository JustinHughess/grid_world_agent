import matplotlib.pyplot as plt
import numpy as np
from environment import GridWorld
from agent import QLearningAgent


def draw_grid(ax, env, agent_pos):
    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
    ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="major", color="#00FF00", linestyle="-", linewidth=2)

    for spine in ax.spines.values():
        spine.set_edgecolor("#00FF00")
        spine.set_linewidth(2)

    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color='#fd0000', alpha=0.7))

    ax.add_patch(plt.Rectangle((env.goal_pos[1], env.goal_pos[0]), 1, 1, color='#00fdfd', alpha=0.7))

    ax.add_patch(plt.Rectangle((agent_pos[1] + 0.15, agent_pos[0] + 0.15), 0.7, 0.7, color='yellow'))


def main():
    print("Grid World Q-Learning Agent")
    print("Yellow=Agent, Cyan=Goal, Red=Obstacles")

    obstacles = [
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14),
        (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14),
    ]
    env = GridWorld(grid_size=15, start_pos=(0, 0), goal_pos=(14, 14), obstacles=obstacles)

    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')

    n_trainings = 1000
    max_steps = 200
    visualize_every = 100
    successes = 0

    for training in range(n_trainings):
        state = env.reset()
        show_this_episode = (training == 0) or ((training + 1) % visualize_every == 0)
        reached_goal = False

        for step in range(max_steps):
            if show_this_episode:
                draw_grid(ax, env, state)
                ax.set_title(f"Training: {training + 1}/{n_trainings}", color='white', fontsize=18)
                plt.pause(0.01)

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

            if state in env.obstacles:
                break

            if done:
                reached_goal = True
                if show_this_episode:
                    draw_grid(ax, env, state)
                    ax.set_title(f"Training: {training + 1}/{n_trainings} - GOAL!", color='white', fontsize=18)
                    plt.pause(0.3)
                break

        if reached_goal:
            successes += 1

        agent.decay_epsilon()

        if (training + 1) % 100 == 0:
            print(f"Training {training + 1}/{n_trainings} - Random moves: {agent.epsilon * 100:.1f}% - Success rate: {successes}%")
            successes = 0

    print("\nTraining complete! Now showing optimal path...")
    plt.pause(1)

    state = env.reset()
    path = [state]
    for step in range(100):
        draw_grid(ax, env, state)
        ax.set_title("Optimal Path (after training)", color='white', fontsize=18)
        plt.pause(0.1)

        action = agent.get_best_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(state)

        if done:
            draw_grid(ax, env, state)
            ax.set_title("Optimal Path - GOAL REACHED!", color='white', fontsize=18)
            print(f"Optimal path found in {step + 1} steps!")
            break

    plt.pause(0.5)
    draw_grid(ax, env, state)
    for pos in path:
        ax.add_patch(plt.Rectangle((pos[1] + 0.1, pos[0] + 0.1), 0.8, 0.8, color="#FFFFFF", alpha=0.7))
    ax.set_title("Final Path Highlighted", color="#FFFFFF", fontsize=18)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
