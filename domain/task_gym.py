import random
from .make_env import make_env
from prettyNEAT import *


class GymTask:
    """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
    """

    def __init__(self, game, paramOnly=False, nReps=1):
        """Initializes task environment

        Args:
          game - (string) - dict key of task to be solved (see domain/config.py)

        Optional:
          paramOnly - (bool)  - only load parameters instead of launching task?
          nReps     - (nReps) - number of trials to get average fitness
        """
        # Network properties
        self.nInput = game.input_size
        self.nOutput = game.output_size
        self.actRange = game.h_act
        self.absWCap = game.weightCap
        self.layers = game.layers
        self.activations = np.r_[np.full(1, 1), game.i_act, game.o_act]

        # Environment
        self.nReps = nReps
        self.maxEpisodeLength = game.max_episode_length
        self.actSelect = game.actionSelect
        if not paramOnly:
            self.env = make_env(game.env_name)

        # == EA-elective-NEAT ==========================================================================================
        self.is_minatar = game.env_name.startswith("minatar:")
        # ==============================================================================================================

        # Special needs...
        self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))

    def getFitness(self, wVec, aVec, hyp=None, view=False, nRep=False, seed=-1):
        """Get fitness of a single individual.

        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          nReps   - (nReps)    - number of trials to get average fitness
          seed    - (int)      - starting random seed for trials

        Returns:
          fitness - (float)    - mean reward over all trials
        """
        if nRep is False:
            nRep = self.nReps
        wVec[np.isnan(wVec)] = 0
        reward = np.empty(nRep)
        for iRep in range(nRep):
            reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed + iRep)
        fitness = np.mean(reward)
        return fitness

    def testInd(self, wVec, aVec, view=False, seed=-1):
        """Evaluate individual on task
        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          seed    - (int)      - starting random seed for trials

        Returns:
          fitness - (float)    - reward earned in trial
        """
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
        # == EA-elective-NEAT ==========================================================================================
            if not self.is_minatar:
                self.env.seed(seed)
        state = self.wrapper_reset()
        # print("wVec: {} ({}),\naVec: {} ({}),\nself.nInput: {},\nself.nOutput: {},\nstate.shape: {}".format(
        #     wVec, wVec.shape, aVec, aVec.shape, self.nInput, self.nOutput, state.shape))
        # ==============================================================================================================
        self.env.t = 0
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)

        # wVec[wVec != 0]
        predName = str(np.mean(wVec[wVec != 0]))
        # == EA-elective-NEAT ==========================================================================================
        # print("annOut: {} ({}),\naction: {} ({})".format(annOut, annOut.shape, action, action.shape))
        state, reward, done, info = self.wrapper_step(action)
        # ==============================================================================================================

        if self.maxEpisodeLength == 0:
            if view:
                self.wrapper_render(done)
            return reward
        else:
            totalReward = reward

        for tStep in range(self.maxEpisodeLength):
            annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)
            # == EA-elective-NEAT ======================================================================================
            state, reward, done, info = self.wrapper_step(action)
            # ==========================================================================================================
            totalReward += reward
            if view:
                self.wrapper_render(done)
            if done:
                break
        return totalReward

    # == EA-elective-NEAT ==============================================================================================
    def wrapper_reset(self):
        if self.is_minatar:
            self.env.reset()
            state = self.env.state()
            state = state.transpose((2, 0, 1))
            state = np.sum([state[i] * (i+1) for i in range(state.shape[0])], axis=0)
            state = state.flatten()
        else:
            state = self.env.reset()
        return state

    def wrapper_step(self, actions):
        if self.is_minatar:
            reward, done = self.env.act(minatar_action(actions))
            state = self.env.state()
            state = state.transpose((2, 0, 1))
            state = np.sum([state[i] * (i+1) for i in range(state.shape[0])], axis=0)
            state = state.flatten()
        else:
            state, reward, done, _ = self.env.step(actions)

        return state, reward, done, {}

    def wrapper_render(self, done):
        if self.is_minatar:
            self.env.display_state(time=50)
            state = self.env.state().transpose((2, 0, 1))
            state = np.sum([state[i] * (i + 1) for i in range(state.shape[0])], axis=0)
            print(state.shape)
        else:
            if self.needsClosed:
                self.env.render(close=done)
            else:
                self.env.render()
    # ==================================================================================================================


# == EA-elective-NEAT ==================================================================================================
def minatar_action(actions):
    actions = actions.flatten()
    action = np.random.choice(np.arange(actions.size), p=actions)
    return action
# ======================================================================================================================
