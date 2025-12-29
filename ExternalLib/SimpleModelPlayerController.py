from AI.GameTranslation.TranslatorFromPolicy import TranslatorFromPolicy
from AI.Models.SimpleModel.Policy import SimpleModel
from Core.Domain.PlayerInput import PlayerInput
from Core.Infrastructure.PlayerController import PlayerController


class SimpleModelPlayerController(PlayerController):
    def __init__(self, player_id, team_id, policy : SimpleModel, translator : TranslatorFromPolicy):
        super().__init__(player_id, team_id)
        self.action = None
        self.policy = policy
        self.translator = translator

    def is_acting(self, state):
        action = self.translator.get_action(self.policy, state)
        for act in action:
            if act > 0.5:
                return True

    def get_action(self, state) -> PlayerInput:
        action = self.translator.get_action(self.policy, state)
        input = PlayerInput(self.team_id, self.player_id, action[0] - action[1], action[2] - action[3] , action[4])
        return input