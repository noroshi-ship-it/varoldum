
import numpy as np
from utils.geometry import toroidal_distance

# Physics constants — physical limits only
MAX_SOCIAL_RANGE = 4.0       # max distance for social force
MAX_ENERGY_TRANSFER = 0.12   # max energy transfer per tick
TRANSFER_EFFICIENCY = 0.9    # thermodynamic loss
MAX_THEFT = 0.15             # max stolen per tick
THEFT_EFFICIENCY = 0.7       # theft is lossy
MAX_COMBAT_DAMAGE = 0.08     # max damage per tick
RETALIATION_FACTOR = 0.6     # Newton's 3rd law


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
        self.inventory_trades = 0

    def resolve_social(self, agents: list, actions: list[dict], tick: int,
                       substance_props=None):
        """
        Single 'social' force channel. Positive = give, negative = take.
        Physics applies: proximity, energy conservation, size.
        Brain decides direction and magnitude. That's it.
        """
        self._tick_interactions = []
        social_rewards = {}
        positive_interactions = {}
        negative_interactions = {}

        action_map = {}
        for agent, action in zip(agents, actions):
            action_map[agent.id] = action

        processed_pairs = set()

        for agent, action in zip(agents, actions):
            if not agent.is_alive:
                continue

            social_force = action.get("social", 0.0)  # [-1, 1] single channel
            signal = action.get("signal", 0.0)

            # No social intent → skip
            if abs(social_force) < 0.001 and signal < 0.001:
                continue

            nearby = []
            for other in agents:
                if other.id == agent.id or not other.is_alive:
                    continue
                dist = toroidal_distance(
                    agent.body.position, other.body.position, self.w, self.h
                )
                if dist < MAX_SOCIAL_RANGE:
                    nearby.append((other, dist))

            if not nearby:
                continue

            for other, dist in nearby:
                proximity = 1.0 - dist / MAX_SOCIAL_RANGE

                # ============================================================
                # SOCIAL FORCE > 0: GIVE (cooperation/sharing)
                # ============================================================
                if social_force > 0:
                    give_force = social_force * proximity
                    available = max(0.0, agent.body.energy - 0.2)
                    share = give_force * available * 0.3
                    share = min(share, MAX_ENERGY_TRANSFER)

                    if share > 0.001:
                        agent.body.energy -= share
                        other.body.energy += share * TRANSFER_EFFICIENCY

                        share_signal = share * 5.0
                        social_rewards[agent.id] = social_rewards.get(agent.id, 0) + share_signal * 0.3
                        social_rewards[other.id] = social_rewards.get(other.id, 0) + share_signal * 0.5
                        positive_interactions[agent.id] = positive_interactions.get(agent.id, 0) + share_signal
                        positive_interactions[other.id] = positive_interactions.get(other.id, 0) + share_signal

                        self._tick_interactions.append(
                            SocialInteraction("cooperate", agent.id, other.id, share, tick)
                        )

                        # Physics consequences: trust/bond updates proportional to transfer
                        trust_delta = share * 3.0
                        if hasattr(agent, 'trust_memory'):
                            agent.trust_memory.update_trust(other.id, trust_delta, tick)
                        if hasattr(other, 'trust_memory'):
                            other.trust_memory.update_trust(agent.id, trust_delta, tick)
                        if hasattr(agent, 'strengthen_bond'):
                            agent.strengthen_bond(other.id, share * 0.4)
                        if hasattr(other, 'strengthen_bond'):
                            other.strengthen_bond(agent.id, share * 0.4)
                        if hasattr(other, 'add_debt'):
                            other.add_debt(agent.id, share * 1.5)
                        if hasattr(agent, '_debts') and other.id in agent._debts:
                            agent._debts[other.id] = max(0, agent._debts[other.id] - share * 2.0)

                # ============================================================
                # SOCIAL FORCE < 0: TAKE (aggression/theft)
                # ============================================================
                elif social_force < 0:
                    take_force = -social_force * proximity  # positive magnitude
                    size_ratio = agent.body.size / max(0.01, other.body.size)
                    take_force *= min(size_ratio, 2.0)

                    if take_force > 0.01:
                        stolen = take_force * 0.2 * other.body.energy
                        stolen = min(stolen, MAX_THEFT)

                        if stolen > 0.001:
                            other.body.energy -= stolen
                            agent.body.energy += stolen * THEFT_EFFICIENCY

                        # Retaliation damage
                        attacker_damage = take_force * MAX_COMBAT_DAMAGE * RETALIATION_FACTOR / max(0.3, size_ratio)
                        defender_damage = take_force * MAX_COMBAT_DAMAGE * min(size_ratio, 2.0)
                        agent.body.take_damage(attacker_damage)
                        other.body.take_damage(defender_damage)

                        social_rewards[agent.id] = social_rewards.get(agent.id, 0) + stolen * 0.5
                        social_rewards[other.id] = social_rewards.get(other.id, 0) - (stolen + defender_damage) * 3.0
                        negative_interactions[other.id] = negative_interactions.get(other.id, 0) + take_force
                        negative_interactions[agent.id] = negative_interactions.get(agent.id, 0) + attacker_damage * 5.0

                        self._tick_interactions.append(
                            SocialInteraction("combat", agent.id, other.id, stolen, tick)
                        )

                        if hasattr(other, 'trust_memory'):
                            other.trust_memory.update_trust(agent.id, -take_force * 0.8, tick)
                        if hasattr(agent, 'trust_memory'):
                            agent.trust_memory.update_trust(other.id, -take_force * 0.3, tick)

                # ============================================================
                # TRADE: signal channel + proximity (both agents need items)
                # ============================================================
                pair_key = (min(agent.id, other.id), max(agent.id, other.id))

                if pair_key not in processed_pairs and signal > 0:
                    trade_force = signal * proximity

                    # Mineral trade
                    if (trade_force > 0.01 and
                            agent.body.mineral_carried > 0.01 and
                            other.body.mineral_carried < agent.body.mineral_carried):
                        trade_amount = trade_force * 0.15
                        trade_amount = min(trade_amount, agent.body.mineral_carried * 0.5)
                        if trade_amount > 0.001:
                            agent.body.mineral_carried -= trade_amount
                            other.body.mineral_carried += trade_amount
                            energy_cost = trade_amount * 0.3
                            other.body.energy -= energy_cost
                            agent.body.energy += energy_cost

                            social_rewards[agent.id] = social_rewards.get(agent.id, 0) + trade_amount * 2.0
                            social_rewards[other.id] = social_rewards.get(other.id, 0) + trade_amount * 1.5
                            positive_interactions[agent.id] = positive_interactions.get(agent.id, 0) + 0.5
                            positive_interactions[other.id] = positive_interactions.get(other.id, 0) + 0.5

                            self._tick_interactions.append(
                                SocialInteraction("trade", agent.id, other.id, trade_amount, tick)
                            )
                            if hasattr(agent, 'trust_memory'):
                                agent.trust_memory.update_trust(other.id, trade_amount * 3.0, tick)
                            if hasattr(other, 'trust_memory'):
                                other.trust_memory.update_trust(agent.id, trade_amount * 3.0, tick)
                            if hasattr(agent, 'strengthen_bond'):
                                agent.strengthen_bond(other.id, trade_amount * 0.3)
                            if hasattr(other, 'strengthen_bond'):
                                other.strengthen_bond(agent.id, trade_amount * 0.3)
                            processed_pairs.add(pair_key)

                    # Inventory trade
                    if (pair_key not in processed_pairs and substance_props is not None and
                            hasattr(agent, 'inventory') and hasattr(other, 'inventory')):
                        trade_will = getattr(agent, '_trade_willingness', 0.5)
                        inv_force = signal * trade_will * proximity
                        if inv_force > 0.01 and agent.inventory.item_count > 0:
                            offer = agent.inventory.best_offer_for(other.inventory, substance_props)
                            if offer is not None:
                                my_slot, their_slot, my_amt, their_amt = offer
                                scale = min(1.0, inv_force * 2.0)
                                my_sid = int(agent.inventory.slots[my_slot, 0])
                                their_sid = int(other.inventory.slots[their_slot, 0])
                                my_removed = agent.inventory.remove_item(my_slot, my_amt * scale)
                                their_removed = other.inventory.remove_item(their_slot, their_amt * scale)
                                if my_removed > 0 and their_removed > 0:
                                    other.inventory.add_item(my_sid, my_removed)
                                    agent.inventory.add_item(their_sid, their_removed)
                                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) + 0.2
                                    social_rewards[other.id] = social_rewards.get(other.id, 0) + 0.2
                                    positive_interactions[agent.id] = positive_interactions.get(agent.id, 0) + 0.5
                                    positive_interactions[other.id] = positive_interactions.get(other.id, 0) + 0.5
                                    if hasattr(agent, 'trust_memory'):
                                        agent.trust_memory.update_trust(other.id, 0.5, tick)
                                    if hasattr(other, 'trust_memory'):
                                        other.trust_memory.update_trust(agent.id, 0.5, tick)
                                    self.inventory_trades += 1
                                    self._tick_interactions.append(
                                        SocialInteraction("inventory_trade", agent.id, other.id, my_removed, tick)
                                    )
                                    processed_pairs.add(pair_key)

        self.interactions.extend(self._tick_interactions)
        if len(self.interactions) > 10000:
            self.interactions = self.interactions[-5000:]

        return social_rewards, positive_interactions, negative_interactions

    @property
    def recent_stats(self) -> dict:
        recent = self._tick_interactions
        if not recent:
            return {"trades": 0, "combats": 0, "cooperations": 0, "inventory_trades": 0}
        return {
            "trades": sum(1 for i in recent if i.type == "trade"),
            "combats": sum(1 for i in recent if i.type == "combat"),
            "cooperations": sum(1 for i in recent if i.type == "cooperate"),
            "inventory_trades": sum(1 for i in recent if i.type == "inventory_trade"),
        }
