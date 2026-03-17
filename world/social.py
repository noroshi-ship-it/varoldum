
import numpy as np
from utils.geometry import toroidal_distance

# Physics constants — these are the only hardcoded values.
# They define physical limits, not decisions.
MAX_SOCIAL_RANGE = 4.0       # max distance for any social force
MAX_ENERGY_TRANSFER = 0.12   # max energy transfer per tick (conservation)
TRANSFER_EFFICIENCY = 0.9    # energy lost in transfer (thermodynamics)
MAX_THEFT = 0.15             # max energy stolen per tick
THEFT_EFFICIENCY = 0.7       # theft is less efficient than cooperation
MAX_COMBAT_DAMAGE = 0.08     # max damage per tick from combat
RETALIATION_FACTOR = 0.6     # defender always fights back


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
        self._tick_interactions = []
        social_rewards = {}
        positive_interactions = {}
        negative_interactions = {}

        # Build action lookup
        action_map = {}
        for agent, action in zip(agents, actions):
            action_map[agent.id] = action

        # Track which pairs already interacted this tick to avoid double-processing
        processed_pairs = set()

        for agent, action in zip(agents, actions):
            if not agent.is_alive:
                continue

            give_intent = action.get("social_give", 0.0)   # [-1, 1] from brain
            take_intent = action.get("social_take", 0.0)    # [-1, 1] from brain
            signal = action.get("signal", 0.0)

            # Find all nearby agents within social range
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
                # Continuous proximity falloff: 1.0 at dist=0, 0.0 at dist=MAX_SOCIAL_RANGE
                proximity = 1.0 - dist / MAX_SOCIAL_RANGE

                # Trust: continuous [0, 1] — modulates all social forces
                trust = 0.5
                if hasattr(agent, 'trust_memory'):
                    trust = agent.trust_memory.get_trust(other.id)

                # Bond strength: continuous [0, 1]
                bond = 0.0
                if hasattr(agent, '_bonds'):
                    bond = agent._bonds.get(other.id, 0.0)

                # Reciprocity urge: continuous [0, 1]
                reciprocity = 0.0
                if hasattr(agent, 'get_reciprocity_urge'):
                    reciprocity = agent.get_reciprocity_urge(other.id)

                # Kinship multiplier: siblings get a boost (physics: shared genes)
                kinship = 1.5 if (agent.parent_id == other.parent_id and agent.parent_id != -1) else 1.0

                # ============================================================
                # ENERGY SHARING (cooperation)
                # Force = give_intent * proximity * trust * kinship
                # Reciprocity adds to give force (debt-driven cooperation)
                # ============================================================
                give_force = max(0.0, give_intent) * proximity * trust * kinship
                give_force += reciprocity * proximity * 0.3  # debt urge

                # Available energy to share (can't share below survival minimum)
                available = max(0.0, agent.body.energy - 0.25)
                share = give_force * available * 0.3
                share = min(share, MAX_ENERGY_TRANSFER)

                if share > 0.001:
                    agent.body.energy -= share
                    other.body.energy += share * TRANSFER_EFFICIENCY

                    # Proportional social signals
                    share_signal = share * 5.0
                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) + share_signal * 0.3
                    social_rewards[other.id] = social_rewards.get(other.id, 0) + share_signal * 0.5
                    positive_interactions[agent.id] = positive_interactions.get(agent.id, 0) + share_signal
                    positive_interactions[other.id] = positive_interactions.get(other.id, 0) + share_signal

                    self._tick_interactions.append(
                        SocialInteraction("cooperate", agent.id, other.id, share, tick)
                    )

                    # Trust grows proportional to amount shared
                    trust_delta = share * 3.0
                    if hasattr(agent, 'trust_memory'):
                        agent.trust_memory.update_trust(other.id, trust_delta, tick)
                    if hasattr(other, 'trust_memory'):
                        other.trust_memory.update_trust(agent.id, trust_delta, tick)

                    # Bonds strengthen
                    bond_delta = share * 0.4
                    if hasattr(agent, 'strengthen_bond'):
                        agent.strengthen_bond(other.id, bond_delta)
                    if hasattr(other, 'strengthen_bond'):
                        other.strengthen_bond(agent.id, bond_delta)

                    # Reciprocity: other now owes agent
                    if hasattr(other, 'add_debt'):
                        other.add_debt(agent.id, share * 1.5)

                    # Pay off own debt
                    if hasattr(agent, '_debts') and other.id in agent._debts:
                        agent._debts[other.id] = max(0, agent._debts[other.id] - share * 2.0)

                # ============================================================
                # AGGRESSION (combat / theft)
                # Force = take_intent * proximity * (1 - trust) * size_advantage
                # Low trust amplifies aggression; high trust suppresses it
                # ============================================================
                size_ratio = agent.body.size / max(0.01, other.body.size)
                take_force = max(0.0, take_intent) * proximity * (1.0 - trust) * min(size_ratio, 2.0)

                if take_force > 0.01:
                    # Energy theft: proportional to force and victim's energy
                    stolen = take_force * 0.2 * other.body.energy
                    stolen = min(stolen, MAX_THEFT)

                    if stolen > 0.001:
                        other.body.energy -= stolen
                        agent.body.energy += stolen * THEFT_EFFICIENCY

                    # Combat damage: both sides take damage, scaled by size
                    attacker_damage = take_force * MAX_COMBAT_DAMAGE * RETALIATION_FACTOR / max(0.3, size_ratio)
                    defender_damage = take_force * MAX_COMBAT_DAMAGE * min(size_ratio, 2.0)
                    agent.body.take_damage(attacker_damage)
                    other.body.take_damage(defender_damage)

                    # Social signals
                    social_rewards[agent.id] = social_rewards.get(agent.id, 0) + stolen * 0.5
                    social_rewards[other.id] = social_rewards.get(other.id, 0) - (stolen + defender_damage) * 3.0
                    negative_interactions[other.id] = negative_interactions.get(other.id, 0) + take_force
                    negative_interactions[agent.id] = negative_interactions.get(agent.id, 0) + attacker_damage * 5.0

                    self._tick_interactions.append(
                        SocialInteraction("combat", agent.id, other.id, stolen, tick)
                    )

                    # Trust destruction: proportional to aggression
                    if hasattr(other, 'trust_memory'):
                        other.trust_memory.update_trust(agent.id, -take_force * 0.8, tick)
                    # Attacker's trust also drops (mutual hostility)
                    if hasattr(agent, 'trust_memory'):
                        agent.trust_memory.update_trust(other.id, -take_force * 0.3, tick)

                # ============================================================
                # MINERAL TRADE (legacy)
                # Force = signal * proximity * trust
                # Both agents must have minerals for exchange to occur
                # ============================================================
                pair_key = (min(agent.id, other.id), max(agent.id, other.id))
                if pair_key not in processed_pairs:
                    trade_force = signal * proximity * trust

                    if (trade_force > 0.01 and
                            agent.body.mineral_carried > 0.05 and
                            other.body.mineral_carried < agent.body.mineral_carried):
                        # Trade amount proportional to force
                        trade_amount = trade_force * 0.15
                        trade_amount = min(trade_amount, agent.body.mineral_carried * 0.5)

                        if trade_amount > 0.001:
                            agent.body.mineral_carried -= trade_amount
                            other.body.mineral_carried += trade_amount
                            # Energy cost/benefit proportional to trade
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
                            if hasattr(other, 'add_debt'):
                                other.add_debt(agent.id, trade_amount)

                            processed_pairs.add(pair_key)

                # ============================================================
                # INVENTORY TRADE
                # Force = signal * trade_willingness * proximity * trust
                # Brain controls signal; genome controls trade_willingness
                # ============================================================
                if (substance_props is not None and
                        hasattr(agent, 'inventory') and hasattr(other, 'inventory') and
                        pair_key not in processed_pairs):
                    trade_will = getattr(agent, '_trade_willingness', 0.5)
                    inv_trade_force = signal * trade_will * proximity * trust

                    if inv_trade_force > 0.01 and agent.inventory.item_count > 0:
                        offer = agent.inventory.best_offer_for(other.inventory, substance_props)
                        if offer is not None:
                            my_slot, their_slot, my_amt, their_amt = offer
                            # Scale trade amounts by force (partial trades possible)
                            scale = min(1.0, inv_trade_force * 2.0)
                            my_amt_scaled = my_amt * scale
                            their_amt_scaled = their_amt * scale

                            my_sid = int(agent.inventory.slots[my_slot, 0])
                            their_sid = int(other.inventory.slots[their_slot, 0])
                            my_removed = agent.inventory.remove_item(my_slot, my_amt_scaled)
                            their_removed = other.inventory.remove_item(their_slot, their_amt_scaled)

                            if my_removed > 0 and their_removed > 0:
                                other.inventory.add_item(my_sid, my_removed)
                                agent.inventory.add_item(their_sid, their_removed)

                                social_rewards[agent.id] = social_rewards.get(agent.id, 0) + 0.2
                                social_rewards[other.id] = social_rewards.get(other.id, 0) + 0.2
                                positive_interactions[agent.id] = positive_interactions.get(agent.id, 0) + 0.5
                                positive_interactions[other.id] = positive_interactions.get(other.id, 0) + 0.5

                                if hasattr(agent, 'trust_memory'):
                                    agent.trust_memory.update_trust(other.id, 0.7, tick)
                                if hasattr(other, 'trust_memory'):
                                    other.trust_memory.update_trust(agent.id, 0.7, tick)

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
