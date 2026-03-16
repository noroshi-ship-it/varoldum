
import numpy as np
from utils.geometry import toroidal_distance


class SocialInteraction:
    __slots__ = ['type', 'agent_a', 'agent_b', 'amount', 'tick']

    def __init__(self, itype: str, a_id: int, b_id: int, amount: float, tick: int):
        self.type = itype
        self.agent_a = a_id
        self.agent_b = b_id
        self.amount = amount
        self.tick = tick


class SocialSystem:

    def __init__(self, world_w: int, world_h: int):
        self.w = world_w
        self.h = world_h
        self.interactions: list[SocialInteraction] = []
        self._tick_interactions: list[SocialInteraction] = []

    def resolve_social(self, agents: list, actions: list[dict], tick: int):
        self._tick_interactions = []
        social_rewards = {}

        positions = {}
        for agent in agents:
            if agent.is_alive:
                positions[agent.id] = agent

        for agent, action in zip(agents, actions):
            if not agent.is_alive:
                continue

            signal_strength = action.get("signal", 0)

            nearby = []
            for other in agents:
                if other.id == agent.id or not other.is_alive:
                    continue
                dist = toroidal_distance(
                    agent.body.position, other.body.position, self.w, self.h
                )
                if dist < 3.0:
                    nearby.append((other, dist))

            if not nearby:
                continue

            nearest, nearest_dist = min(nearby, key=lambda x: x[1])

            if signal_strength > 0.5 and nearest_dist < 2.0:
                if (agent.body.mineral_carried > 0.1 and nearest.body.energy > 0.4 and
                        nearest.body.mineral_carried < agent.body.mineral_carried):
                    trade_amount = 0.1
                    agent.body.mineral_carried -= trade_amount
                    nearest.body.mineral_carried += trade_amount
                    nearest.body.energy -= 0.05
                    agent.body.energy += 0.05
                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) + 0.3
                    social_rewards[nearest.id] = social_rewards.get(nearest.id, 0) + 0.2
                    self._tick_interactions.append(
                        SocialInteraction("trade", agent.id, nearest.id, trade_amount, tick)
                    )

            if signal_strength < 0.1 and agent.body.energy < 0.3 and nearest_dist < 1.5:
                attack_success = agent.body.size > nearest.body.size * 0.8
                if attack_success:
                    stolen = min(0.1, nearest.body.energy * 0.2)
                    nearest.body.energy -= stolen
                    agent.body.energy += stolen * 0.7
                    nearest.body.take_damage(0.05)
                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) + 0.1
                    social_rewards[nearest.id] = social_rewards.get(nearest.id, 0) - 0.5
                    self._tick_interactions.append(
                        SocialInteraction("combat", agent.id, nearest.id, stolen, tick)
                    )
                else:
                    agent.body.take_damage(0.08)
                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) - 0.3
                    self._tick_interactions.append(
                        SocialInteraction("combat_fail", agent.id, nearest.id, 0, tick)
                    )

            if (agent.parent_id == nearest.parent_id and agent.parent_id != -1 and
                    signal_strength > 0.3 and agent.body.energy > 0.5 and
                    nearest.body.energy < 0.3):
                share = min(0.08, agent.body.energy - 0.4)
                if share > 0:
                    agent.body.energy -= share
                    nearest.body.energy += share * 0.9
                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) + 0.2
                    social_rewards[nearest.id] = social_rewards.get(nearest.id, 0) + 0.3
                    self._tick_interactions.append(
                        SocialInteraction("cooperate", agent.id, nearest.id, share, tick)
                    )

        self.interactions.extend(self._tick_interactions)
        if len(self.interactions) > 10000:
            self.interactions = self.interactions[-5000:]

        return social_rewards

    @property
    def recent_stats(self) -> dict:
        recent = self._tick_interactions
        if not recent:
            return {"trades": 0, "combats": 0, "cooperations": 0}
        return {
            "trades": sum(1 for i in recent if i.type == "trade"),
            "combats": sum(1 for i in recent if i.type in ("combat", "combat_fail")),
            "cooperations": sum(1 for i in recent if i.type == "cooperate"),
        }
