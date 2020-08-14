import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import PlottingConstants


class AlgorithmLogger:
    """
        time_steps:         number of time steps each episode lasted
        scores:             obtained reward at each episode
        episode:            number of episodes ran
        actions:            action taken at each time_step for each episode
        action:             number of actions taken in current episode
        last_no_of_actions: total number of actions taken
                            in the previous episode
        states:             full state for each time step
                            in each of the episodes
        successes:          indicates whether the episode was successful or not
    """

    def __init__(self, max_no_of_episodes, max_time_steps_per_episode=1000,
                 observation_size=4, goal=1.0, start=-1.25):
        self.time_steps = np.zeros(shape=(max_no_of_episodes,), dtype=np.int)
        self.scores = np.zeros(shape=(max_no_of_episodes,))
        self.episode = 0

        self.actions = np.full(shape=(max_no_of_episodes,
                                      max_time_steps_per_episode),
                               fill_value=np.nan,
                               dtype=np.float)

        self.action, self.last_no_of_actions = 0, -1

        self.states = np.full(shape=(max_no_of_episodes,
                                     max_time_steps_per_episode,
                                     observation_size),
                              fill_value=np.nan,
                              dtype=np.float)

        self.successes = np.full(shape=(max_no_of_episodes,),
                                 fill_value=False, dtype=np.bool)

        self.start = start
        self.goal = goal

    def account_state_action(self, force, state):
        self.actions[self.episode, self.action] = force
        self.states[self.episode, self.action, :] = state
        self.action += 1

    def account_whole_episode(self, length, reward, success=False):
        self.time_steps[self.episode] = length
        self.scores[self.episode] = reward
        self.successes[self.episode] = success
        self.episode += 1
        self.last_no_of_actions = self.action
        self.action = 0

    def print_statistics(self):
        # assert self.episode > 0
        if self.episode == 0:  # TODO
            return

        no_successful_episodes = np.sum(self.successes)
        ratio_of_successes = no_successful_episodes / self.episode
        print(f'SUCCESSFUL episodes: {no_successful_episodes}, '
              f'UNSUCCESSFUL episodes: {self.episode - no_successful_episodes}')
        print(f'Success ratio = {ratio_of_successes}')

        print('All rewards statistics:')
        print(pd.Series(self.scores[0:self.episode]).describe())

        print('All episode length statistics:')
        print(pd.Series(self.time_steps[0:self.episode]).describe())

        if np.sum(self.successes) > 0:
            print('Successful rewards statistics:')
            print(pd.Series(self.scores[self.successes]).describe())
            print('Successful episode length statistics:')
            print(pd.Series(self.time_steps[self.successes]).describe())

    def plot_whole_experiment_summary(self, what):

        marker = ''
        color = np.array(list(map(lambda s: 'tab:green' if s else 'tab:red',
                                  self.successes[:self.episode])))

        x_values = np.arange(self.episode)

        if what == PlottingConstants.EPISODE_TIME_STEPS:
            y_values = self.time_steps[0:self.episode]
            title = 'Dolžina epizode'
            y_label = 'Korak'
            x_label = 'Epizoda'
            plt.scatter(x_values, y_values, marker=marker, color=color)
        elif what == PlottingConstants.EPISODE_REWARD:
            y_values = self.scores[0:self.episode]
            title = 'Dobljena nagrada'
            y_label = 'Nagrada'
            x_label = 'Epizoda'
            plt.scatter(x_values, y_values, marker=marker, color=color)
        elif what == PlottingConstants.MINIMAL_DISTANCE_PER_EPISODE:
            y_values = np.nanmin(np.abs(
                self.states[0:self.episode, :, 0] - self.goal), axis=1)
            plt.ylim(0, 2 * np.pi)
            plt.yticks([k * np.pi for k in range(0, 2 + 1)],
                       ['0', '$\pi$', '$2\pi$'])
            title = 'Najmanjša dosežena razdalja na epizodo'
            y_label = 'Razdalja'
            x_label = 'Epizoda'
            plt.scatter(x_values, y_values, marker=marker, color=color)
        elif what == PlottingConstants.AVERAGE_DISTANCE_PER_EPISODE:
            y_values = np.nanmean(np.abs(
                self.states[0:self.episode, :, 0] - self.goal), axis=1)
            plt.ylim(0, 2 * np.pi)
            plt.yticks([k * np.pi for k in range(0, 2 + 1)],
                       ['0', '$\pi$', '$2\pi$'])
            title = 'Povprečna razdalja na epizodo'
            y_label = 'Razdalja'
            x_label = 'Epizoda'
            plt.scatter(x_values, y_values, marker=marker, color=color)
        elif what == PlottingConstants.NUMBER_OF_FAILURES:
            y_values = np.cumsum(-1 * self.successes[0:self.episode] + 1)
            title = 'Število neuspehov v poskusu'
            y_label = 'Število'
            x_label = 'Epizoda'
            color = 'tab:orange'
            plt.plot(y_values, linestyle='', marker=marker, color=color)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def plot_successful_episodes_experiment_summary(self, what):

        marker = ''
        x_values = np.arange(np.sum(self.successes))

        if what == PlottingConstants.EPISODE_TIME_STEPS_SUCCESSFUL:
            y_values = self.time_steps[self.successes]
            title = 'Dolžine uspešnih epizod'
            y_label = 'Korak'
            x_label = 'Epizoda'
            color = 'lime'
        elif what == PlottingConstants.EPISODE_REWARD_SUCCESSFUL:
            y_values = self.scores[self.successes]
            title = 'Nagrade v uspešnih epizodah'
            y_label = 'Nagrada'
            x_label = 'Epizoda'
            color = 'lime'
        elif what == PlottingConstants.MINIMAL_DISTANCE_PER_EPISODE_SUCCESSFUL:
            y_values = np.nanmin(np.abs(
                self.states[self.successes, :, 0] - self.goal), axis=1)
            plt.ylim(0, 2 * np.pi)
            plt.yticks([k * np.pi for k in range(0, 2 + 1)],
                       ['0', '$\pi$', '$2\pi$'])
            title = 'Najkrajša dosežena razdalja do cilja v uspešnih epizodah'
            y_label = 'Razdalja'
            x_label = 'Epizoda'
            color = 'lime'
        elif what == PlottingConstants.AVERAGE_DISTANCE_PER_EPISODE_SUCCESSFUL:
            y_values = np.nanmean(np.abs(
                self.states[self.successes, :, 0] - self.goal), axis=1)
            plt.ylim(0, 2 * np.pi)
            plt.yticks([k * np.pi for k in range(0, 2 + 1)],
                       ['0', '$\pi$', '$2\pi$'])
            title = 'Povprečna dosežena razdalja do cilja v uspešnih epizodah'
            y_label = 'Razdalja'
            x_label = 'Epizoda'
            color = 'lime'

        plt.scatter(x_values, y_values, marker=marker, color=color)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def plot_averaged_episodes_summary(self, what):

        color = 'tab:blue'
        line_style = 'solid'
        marker = ''

        if what == PlottingConstants.AVERAGE_ACTION:
            y_values = np.nanmean(self.actions[0:self.episode, :], axis=0)
            title = 'Povprečna akcija na vsakem koraku epizode'
            y_label = 'Akcija'
            x_label = 'Korak'
            color = 'tab:pink'
            line_style = ''
        elif what == PlottingConstants.AVERAGE_LOCATION:
            y_values = np.nanmean(self.states[0:self.episode, :, 0], axis=0)
            title = 'Povprečna lokacija na vsakem koraku epizode'
            y_label = 'Lokacija'
            x_label = 'Korak'
            # plt.ylim(self.start - 0.5, self.goal + 0.5)
            # plt.yticks([k * np.pi for k in range(-1, 1 + 1)],
            #            ['$-\pi$', '0', '$\pi$'])
            plt.plot(
                np.ones(shape=(len(y_values),)) * self.start,
                linestyle='solid',
                color='crimson')  # start
            plt.plot(np.ones(shape=(len(y_values),)) * self.goal,
                     linestyle='solid',
                     color='limegreen')  # goal
        elif what == PlottingConstants.AVERAGE_DISTANCE:
            y_values = np.nanmean(np.abs(
                self.states[0:self.episode, :, 0] - self.goal), axis=0)
            title = 'Povprečna razdalja od cilja na vsakem koraku epizode'
            y_label = 'Razdalja'
            x_label = 'Korak'
            color = 'tab:cyan'
            # plt.ylim(0, 2 * np.pi)
            # plt.yticks([k * np.pi for k in range(0, 2 + 1)],
            #            ['0', '$\pi$', '$2\pi$'])
            plt.plot(y_values, linestyle=line_style, color=color)
        elif what == PlottingConstants.AVERAGE_LINEAR_VELOCITY:
            y_values = np.nanmean(self.states[0:self.episode, :, 1], axis=0)
            title = 'Povprečna linearna hitrost na vsakem koraku epizode'
            y_label = 'Hitrost'
            x_label = 'Korak'
        elif what == PlottingConstants.AVERAGE_ANGLE:
            y_values = np.nanmean(self.states[0:self.episode, :, 2], axis=0)
            title = 'Povprečen kot palice na vsakem koraku epizode'
            y_label = 'Kot'
            x_label = 'Korak'
        elif what == PlottingConstants.AVERAGE_ANGULAR_VELOCITY:
            y_values = np.nanmean(self.states[0:self.episode, :, 3], axis=0)
            title = 'Povprečna kotna hitrost na vsakem koraku epizode'
            y_label = 'Hitrost'
            x_label = 'Korak'
        elif what == PlottingConstants.AVERAGE_HEIGHT:
            y_values = np.nanmean(self.states[0:self.episode, :, 4], axis=0)
            title = 'Povprečna višina na vsakem koraku epizode'
            y_label = 'Višina'
            x_label = 'Korak'
            color = 'tab:green'

        plt.plot(y_values, linestyle=line_style, marker=marker, color=color)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def plot_averaged_successful_episodes_summary(self, what):

        color = 'tab:blue'
        line_style = 'solid'
        marker = ''

        if what == PlottingConstants.AVERAGE_ACTION_SUCCESSFUL:
            y_values = np.mean(self.actions[self.successes, :], axis=0)
            title = 'Povprečna akcija na vsakem koraku uspešne epizode '
            y_label = 'Akcija'
            x_label = 'Korak'
            color = 'tab:pink'
            line_style = ''
        elif what == PlottingConstants.AVERAGE_LOCATION_SUCCESSFUL:
            y_values = np.nanmean(self.states[self.successes, :, 0], axis=0)
            title = 'Povprečna lokacija na vsakem koraku uspešne epizode'
            y_label = 'Lokacija'
            x_label = 'Korak'
            plt.ylim(-np.pi, np.pi)
            plt.yticks([k * np.pi for k in range(-1, 1 + 1)],
                       ['$-\pi$', '0', '$\pi$'])
            plt.plot(
                np.ones(shape=(len(y_values),)) * (-3 / 4 * np.pi),
                linestyle='solid',
                color='crimson')  # start
            plt.plot(np.ones(shape=(len(y_values),)) * self.goal,
                     linestyle='solid',
                     color='limegreen')  # goal
        elif what == PlottingConstants.AVERAGE_DISTANCE_SUCCESSFUL:
            y_values = np.nanmean(np.abs(
                self.states[self.successes, :, 0] - self.goal), axis=0)
            title = 'Povprečna razdalja od cilja na vsakem koraku ' + \
                'uspešne epizode'
            y_label = 'Razdalja'
            x_label = 'Korak'
            color = 'tab:cyan'
            plt.ylim(0, 2 * np.pi)
            plt.yticks([k * np.pi for k in range(0, 2 + 1)],
                       ['0', '$\pi$', '$2\pi$'])
            plt.plot(y_values, linestyle=line_style, color=color)
        elif what == PlottingConstants.AVERAGE_LINEAR_VELOCITY_SUCCESSFUL:
            y_values = np.nanmean(self.states[self.successes, :, 1], axis=0)
            title = 'Povprečna linearna hitrost na vsakem koraku ' + \
                'uspešne epizode'
            y_label = 'Hitrost'
            x_label = 'Korak'
        elif what == PlottingConstants.AVERAGE_ANGLE_SUCCESSFUL:
            y_values = np.nanmean(self.states[self.successes, :, 2], axis=0)            
            title = 'Povprečen kot palice na vsakem koraku uspešne epizode'
            y_label = 'Kot'
            x_label = 'Korak'
        elif what == PlottingConstants.AVERAGE_ANGULAR_VELOCITY_SUCCESSFUL:
            y_values = np.mean(self.states[self.successes, :, 3], axis=0)
            title = 'Povprečna kotna hitrost na vsakem koraku uspešne epizode'
            y_label = 'Hitrost'
            x_label = 'Korak'

        plt.plot(y_values, linestyle=line_style, marker=marker, color=color)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def plot_single_episodes_summary(self, what):

        color = 'tab:blue'
        line_style = 'solid'
        marker = ''

        if what == PlottingConstants.ACTION:
            y_values = self.actions[self.episode - 1,
                       0:self.time_steps[self.episode - 1]]
            title = 'Akcija na vsakem koraku epizode'
            y_label = 'Akcija'
            x_label = 'Korak'
            color = 'tab:pink'
            line_style = ''
        elif what == PlottingConstants.LOCATION:
            y_values = self.states[self.episode - 1,
                       0:self.time_steps[self.episode - 1], 0]
            title = 'Lokacija na vsakem koraku epizode'
            y_label = 'Lokacija'
            x_label = 'Korak'
            # plt.ylim(-np.pi, np.pi)
            # plt.yticks([k * np.pi for k in range(-1, 1 + 1)],
            #            ['$-\pi$', '0', '$\pi$'])
            plt.plot(
                np.ones(shape=(self.time_steps[self.episode - 1],)) * self.start,
                linestyle='solid',
                color='crimson')  # start
            plt.plot(np.ones(
                shape=(self.time_steps[self.episode - 1],)) * self.goal,
                     linestyle='solid',
                     color='limegreen')  # goal
            plt.plot(y_values, linestyle=line_style, color=color)
        elif what == PlottingConstants.DISTANCE:
            y_values = np.abs(
                self.states[self.episode - 1,
                0:self.time_steps[self.episode - 1], 0] - self.goal)
            title = 'Razdalja od cilja na vsakem koraku epizode'
            y_label = 'Razdalja'
            x_label = 'Korak'
            color = 'tab:cyan'
            plt.ylim(0, 2 * np.pi)
            plt.yticks([k * np.pi for k in range(0, 2 + 1)],
                       ['0', '$\pi$', '$2\pi$'])
        elif what == PlottingConstants.LINEAR_VELOCITY:
            y_values = self.states[self.episode - 1,
                       0:self.time_steps[self.episode - 1], 1]
            title = 'Linearna hitrost na vsakem koraku epizode'
            y_label = 'Hitrost'
            x_label = 'Korak'
        elif what == PlottingConstants.ANGLE:
            y_values = self.states[self.episode - 1,
                       0:self.time_steps[self.episode - 1], 2]
            title = 'Kot palice na vsakem koraku epizode'
            y_label = 'Kot'
            x_label = 'Korak'
        elif what == PlottingConstants.ANGULAR_VELOCITY:
            y_values = self.states[self.episode - 1,
                       0:self.time_steps[self.episode - 1], 3]
            title = 'Kotna hitrost na vsakem koraku epizode'
            y_label = 'Hitrost'
            x_label = 'Korak'
        elif what == PlottingConstants.HEIGHT:
            y_values = np.nanmean(self.states[0:self.episode, :, 4], axis=0)
            title = 'Višina na vsakem koraku epizode'
            y_label = 'Višina'
            x_label = 'Korak'
            color = 'tab:green'


        plt.plot(y_values, linestyle=line_style, marker=marker, color=color)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def plot_summary(self, what=PlottingConstants.EPISODE_TIME_STEPS):
        # assert self.episode > 0
        if self.episode == 0:  # TODO
            return

        # WHOLE EXPERIMENT STATISTICS
        if what in PlottingConstants.WHOLE_RUN_STATISTICS:
            self.plot_whole_experiment_summary(what)
        # WHOLE EXPERIMENT STATISTICS OF SUCCESSFUL EPISODES
        elif what in PlottingConstants.AVERAGE_SUCCESSFUL_EPISODE_EXPERIMENT_STATISTICS:
            self.plot_successful_episodes_experiment_summary(what)
        # AVERAGED EPISODE STATISTICS
        elif what in PlottingConstants.AVERAGE_EPISODE_STATISTICS:
            self.plot_averaged_episodes_summary(what)
        # AVERAGED SUCCESSFUL EPISODE STATISTICS
        elif what in PlottingConstants.AVERAGE_SUCCESSFUL_EPISODE_STATISTICS:
            self.plot_averaged_successful_episodes_summary(what)
        # SINGLE EPISODE STATISTICS
        elif what in PlottingConstants.SINGLE_EPISODE_STATISTICS:
            self.plot_single_episodes_summary(what)
