from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Models.SimpleModel.TranslatorToPolicyImpl import TranslatorToPolicyImpl
from AI.Models.SimpleModel.TranslatorFromPolicyImpl import TranslatorFromPolicyImpl
from Core.Infrastructure.Persistence.ArrayGameRepository import ArrayGameRepository
from Core.Infrastructure.Physics.PhysicsImpl import PhysicsImpl
from Core.Presentation.GamePresentation.DrawingImpl import DrawingImpl
from ExternalLib.PygameDrawingInterfaceImpl import PygameDrawingInterfaceImpl
from ExternalLib.PygamePlayerController import PygamePlayerController
from ExternalLib.SimpleModelPlayerController import SimpleModelPlayerController
from Main import CreatingGameConfigs, CreatingDrawingConfigs
from Main.GameLoop import GameLoop

repo = ArrayGameRepository()
physics = PhysicsImpl(CreatingGameConfigs.game_config.physics)
external_drawing = PygameDrawingInterfaceImpl(CreatingDrawingConfigs.drawing_config.window_size, CreatingDrawingConfigs.drawing_config.colors.background_color)

drawing = DrawingImpl(external_drawing, CreatingGameConfigs.game_config, CreatingDrawingConfigs.drawing_config)

game_loop = GameLoop(repo, physics, CreatingGameConfigs.game_config, drawing)

player_controller = PygamePlayerController(game_loop.players_blue_ids[0], 1)

# ai player
translator_from = TranslatorFromPolicyImpl()
translator_to = TranslatorToPolicyImpl(CreatingGameConfigs.game_config, game_loop.players_red_ids[0], 0)

policy = SimpleModel(translator_to)

ai_player_controller = SimpleModelPlayerController(game_loop.players_red_ids[0], 0, policy, translator_from)

game_loop.set_player_controller(ai_player_controller)
game_loop.set_player_controller(player_controller)

game_loop.run()