import torch
from Core.Objects.Map import Map
import Constants
from Player_actions.Net_action import Net_action

class Environment:
    def __init__(self, nn1, nn2, num_steps = 1024):
        self.num_steps = num_steps
        self.nn1 = nn1
        self.nn2 = nn2
        self.map = Map()

        #TODO исправить перебор игроков

        self.count = 0

        self.net_action_team1 = Net_action(self.map, nn1, self.map.players_team1[0], is_team_1=True)
        self.net_action_team2 = Net_action(self.map, nn2, self.map.players_team2[0], is_team_1=False)

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

    def step(self, action1, action2):

        self.net_action_team1.act(action1)
        self.net_action_team2.act(action2)
    
        self.map.move_balls()

        self.count += 1

        r1, r2, natural_done = self.improved_rewards()

         # === ОПРЕДЕЛЯЕМ truncated ===
        truncated = self.count >= (self.num_steps)
        
        # Эпизод завершён, если естественный конец ИЛИ таймаут
        done = natural_done or truncated
        
        info = {'truncated': truncated, 'natural_done': natural_done}
        
        if natural_done:
            pass

        if truncated:
            pass


        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2), (r1, r2), done, info

    def reset(self):
        self.count = 0
        self.score_team1 = 0
        self.score_team2 = 0

        self.map.load_random()

        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        d1 = player1.position - ball.position
        sqrt1 = torch.linalg.vector_norm(d1)
        dist1 = max(0,sqrt1 - player1.radius - ball.radius)
        
        d2 = player2.position - ball.position
        sqrt2 = torch.linalg.vector_norm(d2)
        dist2 = max(0,sqrt2 - player2.radius - ball.radius)

        ball = self.map.balls[0]
        goal_right = torch.tensor([Constants.right_gates_center_x, Constants.y_center])
        goal_left = torch.tensor([Constants.left_gates_center_x, Constants.y_center])

        dist_team1_to_target = torch.linalg.vector_norm(ball.position - goal_right)
        dist_team2_to_target = torch.linalg.vector_norm(ball.position - goal_left)

        # === 2. COMPUTE POTENTIALS INCLUDING ALL SIGNALS ===
        # Goal positions
        goal_team1_opponent = torch.tensor([Constants.right_gates_center_x, Constants.y_center])
        goal_team2_opponent = torch.tensor([Constants.left_gates_center_x, Constants.y_center])

        # Distances to opponent goals (normalized)
        max_dist = torch.linalg.vector_norm(torch.tensor(Constants.window_size, dtype=torch.float32))
        dist_to_opp_goal_team1 = torch.linalg.vector_norm(ball.position - goal_team1_opponent)
        dist_to_opp_goal_team2 = torch.linalg.vector_norm(ball.position - goal_team2_opponent)

        pot_goal_team1 = -dist_to_opp_goal_team1 / max_dist
        pot_goal_team2 = -dist_to_opp_goal_team2 / max_dist

        # Player activity potential (normalized speed, penalize inactivity)
        max_speed = 0.5  # example max player speed
        pot_activity_team1 = (player1.velocity.norm() / max_speed) * 0.05
        pot_activity_team2 = (player2.velocity.norm() / max_speed) * 0.05

        # Boundary penalty potential: strong outside, ramps near edges to keep agents in
        def boundary_potential(player):
            x, y = player.position
            w, h = Constants.window_size
            # If outside field — strong penalty
            if not (0 <= x <= w) or not (0 <= y <= h):
                return -3.0
            # Inside: if close to border, ramp negative potential
            margin = 50.0  # px margin where we start to penalize
            dist_to_edges = torch.tensor([x, y, w - x, h - y])
            min_dist = torch.min(dist_to_edges)
            if min_dist < margin:
                return -1.5 * (1 - min_dist / margin)
            return 0.0

        pot_boundary_team1 = boundary_potential(player1)
        pot_boundary_team2 = boundary_potential(player2)

        pot_dist_to_ball_team1 = (-(player1.position - ball.position).norm() /  (2 * Constants.window_size[0]))
        pot_dist_to_ball_team2 = (-(player2.position - ball.position).norm() / (2 * Constants.field_size[0]))

        ball_vel_to_goal1 = torch.dot(ball.velocity, goal_team1_opponent - ball.position) / (max_dist * max_speed + 1e-6)
        ball_vel_to_goal2 = torch.dot(ball.velocity, goal_team2_opponent - ball.position) / (max_dist * max_speed + 1e-6)
        ball_vel_to_goal1 = torch.clamp(ball_vel_to_goal1, -2.0, 2.0)
        ball_vel_to_goal2 = torch.clamp(ball_vel_to_goal2, -2.0, 2.0)

        # Aggregate total potential for each team:
        total_potential_team1 = (
            5 * pot_goal_team1 +
            3 * ball_vel_to_goal1 +
            pot_activity_team1 +
            pot_dist_to_ball_team1 +
            3 * pot_boundary_team1
        )

        total_potential_team2 = (
            5 * pot_goal_team2 +
            3 * ball_vel_to_goal2 +
            pot_activity_team2 +
            pot_dist_to_ball_team2 +
            3 * pot_boundary_team2
        )

        self.last_potential_team1 = total_potential_team1
        self.last_potential_team2 = total_potential_team2


        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2)
    

    def improved_rewards(self):
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        r_team1 = 0.0
        r_team2 = 0.0
        done = False

        # === 1. GOAL REWARDS (HIGH PRIORITY, SPARSE) ===
        if self.map.score_team1:
            r_team1 += 70.0
            r_team2 -= 70.0
            done = True
            return torch.tensor([r_team1]), torch.tensor([r_team2]), done

        if self.map.score_team2:
            r_team2 += 70.0
            r_team1 -= 70.0
            done = True
            return torch.tensor([r_team1]), torch.tensor([r_team2]), done

        # === 2. COMPUTE POTENTIALS INCLUDING ALL SIGNALS ===
        # Goal positions
        goal_team1_opponent = torch.tensor([Constants.right_gates_center_x, Constants.y_center])
        goal_team2_opponent = torch.tensor([Constants.left_gates_center_x, Constants.y_center])

        # Distances to opponent goals (normalized)
        max_dist = torch.linalg.vector_norm(torch.tensor(Constants.window_size, dtype=torch.float32))
        dist_to_opp_goal_team1 = torch.linalg.vector_norm(ball.position - goal_team1_opponent)
        dist_to_opp_goal_team2 = torch.linalg.vector_norm(ball.position - goal_team2_opponent)

        pot_goal_team1 = -dist_to_opp_goal_team1 / max_dist
        pot_goal_team2 = -dist_to_opp_goal_team2 / max_dist

        max_speed = 0.5  # example max player speed

        # Скорость мяча в сторону нужных ворот (поощряет удары в правильном направлении)
        ball_vel_to_goal1 = torch.dot(ball.velocity, goal_team1_opponent - ball.position) / (max_dist * max_speed + 1e-6)
        ball_vel_to_goal2 = torch.dot(ball.velocity, goal_team2_opponent - ball.position) / (max_dist * max_speed + 1e-6)
        ball_vel_to_goal1 = torch.clamp(ball_vel_to_goal1, -2.0, 2.0)
        ball_vel_to_goal2 = torch.clamp(ball_vel_to_goal2, -2.0, 2.0)


        # Player activity potential (normalized speed, penalize inactivity)
        pot_activity_team1 = (player1.velocity.norm() / max_speed) * 0.05
        pot_activity_team2 = (player2.velocity.norm() / max_speed) * 0.05

        # Boundary penalty potential (large negative potential if out of bounds)
        def boundary_potential(player):
            x, y = player.position
            w, h = Constants.window_size
            # If outside field — strong penalty
            if not (0 <= x <= w) or not (0 <= y <= h):
                return -3.0
            # Inside: if close to border, ramp negative potential
            margin = 50.0  # px margin where we start to penalize
            dist_to_edges = torch.tensor([x, y, w - x, h - y])
            min_dist = torch.min(dist_to_edges)
            if min_dist < margin:
                return -1.5 * (1 - min_dist / margin)
            return 0.0

        pot_boundary_team1 = boundary_potential(player1)
        pot_boundary_team2 = boundary_potential(player2)

        pot_dist_to_ball_team1 = (-(player1.position - ball.position).norm() / (2 * Constants.window_size[0]))
        pot_dist_to_ball_team2 = (-(player2.position - ball.position).norm() / (2 * Constants.window_size[0]))

        # Aggregate total potential for each team:
        total_potential_team1 = (
            5 * pot_goal_team1 +
            3 * ball_vel_to_goal1 +
            pot_activity_team1 +
            pot_dist_to_ball_team1 +
            3 * pot_boundary_team1
        )

        total_potential_team2 = (
            5 * pot_goal_team2 +
            3 * ball_vel_to_goal2 +
            pot_activity_team2 +
            pot_dist_to_ball_team2 +
            3 * pot_boundary_team2
        )

        # === 3. CALCULATE SHAPING REWARD AS DIFFERENCE OF POTENTIALS ===
        gamma = 0.99  # discount factor, should match PPO gamma
        shaping_team1 = gamma * total_potential_team1 - self.last_potential_team1
        shaping_team2 = gamma * total_potential_team2 - self.last_potential_team2

        # Scale shaping to balance sparse rewards (adjust factor as needed)
        scale_factor = 1
        r_team1 += shaping_team1 * scale_factor
        r_team2 += shaping_team2 * scale_factor

        # === 4. UPDATE LAST POTENTIALS FOR NEXT STEP ===
        self.last_potential_team1 = total_potential_team1
        self.last_potential_team2 = total_potential_team2

        return torch.tensor([r_team1]), torch.tensor([r_team2]), done





                    
                