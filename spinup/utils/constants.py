class PlottingConstants:
    EPISODE_TIME_STEPS = 'episode_time_steps'
    EPISODE_REWARD = 'episode_reward'
    MINIMAL_DISTANCE_PER_EPISODE = 'minimal_distance_per_episode'
    AVERAGE_DISTANCE_PER_EPISODE = 'average_distance_per_episode'
    NUMBER_OF_FAILURES = 'number_of_failures'

    WHOLE_RUN_STATISTICS = [EPISODE_TIME_STEPS, EPISODE_REWARD,
                            MINIMAL_DISTANCE_PER_EPISODE,
                            AVERAGE_DISTANCE_PER_EPISODE,
                            NUMBER_OF_FAILURES]

    EPISODE_TIME_STEPS_SUCCESSFUL = 'episode_time_steps_successful'
    EPISODE_REWARD_SUCCESSFUL = 'episode_reward_successful'
    MINIMAL_DISTANCE_PER_EPISODE_SUCCESSFUL = \
        'minimal_distance_per_episode_successful'
    AVERAGE_DISTANCE_PER_EPISODE_SUCCESSFUL = \
        'average_distance_per_episode_successful'

    AVERAGE_SUCCESSFUL_EPISODE_EXPERIMENT_STATISTICS = [
        EPISODE_TIME_STEPS_SUCCESSFUL,
        EPISODE_REWARD_SUCCESSFUL,
        MINIMAL_DISTANCE_PER_EPISODE_SUCCESSFUL,
        AVERAGE_DISTANCE_PER_EPISODE_SUCCESSFUL]

    AVERAGE_ACTION_SUCCESSFUL = 'average_action_successful'
    AVERAGE_LOCATION_SUCCESSFUL = 'average_location_successful'
    AVERAGE_DISTANCE_SUCCESSFUL = 'average_distance_successful'
    AVERAGE_LINEAR_VELOCITY_SUCCESSFUL = 'average_linear_velocity_successful'
    AVERAGE_ANGLE_SUCCESSFUL = 'average_angle_successful'
    AVERAGE_ANGULAR_VELOCITY_SUCCESSFUL = 'average_angular_velocity_successful'

    AVERAGE_SUCCESSFUL_EPISODE_STATISTICS = [AVERAGE_ACTION_SUCCESSFUL,
                                             AVERAGE_LOCATION_SUCCESSFUL,
                                             AVERAGE_DISTANCE_SUCCESSFUL,
                                             AVERAGE_LINEAR_VELOCITY_SUCCESSFUL,
                                             AVERAGE_ANGLE_SUCCESSFUL,
                                             AVERAGE_ANGULAR_VELOCITY_SUCCESSFUL]

    AVERAGE_ACTION = 'average_action'
    AVERAGE_LOCATION = 'average_location'
    AVERAGE_DISTANCE = 'average_distance'
    AVERAGE_LINEAR_VELOCITY = 'average_linear_velocity'
    AVERAGE_ANGLE = 'average_angle'
    AVERAGE_ANGULAR_VELOCITY = 'average_angular_velocity'
    AVERAGE_HEIGHT = 'average_height'

    AVERAGE_EPISODE_STATISTICS = [AVERAGE_ACTION,
                                  AVERAGE_LOCATION,
                                  AVERAGE_DISTANCE,
                                  AVERAGE_LINEAR_VELOCITY,
                                  AVERAGE_ANGLE,
                                  AVERAGE_ANGULAR_VELOCITY]

    ACTION = 'action'
    LOCATION = 'location'
    DISTANCE = 'distance'
    LINEAR_VELOCITY = 'linear_velocity'
    ANGLE = 'angle'
    ANGULAR_VELOCITY = 'angular_velocity'
    HEIGHT = 'height'

    SINGLE_EPISODE_STATISTICS = [ACTION,
                                 LOCATION,
                                 DISTANCE,
                                 LINEAR_VELOCITY,
                                 ANGLE,
                                 ANGULAR_VELOCITY]

    WHOLE_EXPERIMENT_REWARDS = 'whole_experiment_rewards'
    WHOLE_EXPERIMENT_TIME_STEPS = 'whole_experiment_time_steps'

    WHOLE_EXPERIMENT_STATISTICS = [WHOLE_EXPERIMENT_REWARDS,
                                   WHOLE_EXPERIMENT_TIME_STEPS]
