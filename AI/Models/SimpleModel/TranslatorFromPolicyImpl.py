from AI.GameTranslation.TranslatorFromPolicy import TranslatorFromPolicy
from AI.Models.SimpleModel.Policy import SimpleModel


class TranslatorFromPolicyImpl(TranslatorFromPolicy):
    def get_action(self, policy : SimpleModel, state):
        action, _, _ = policy.get_action(state)
        return action.tolist()