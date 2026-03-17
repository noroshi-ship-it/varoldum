
import numpy as np


class Inventory:
    """Discrete item inventory for agents. Items are chemistry substances."""

    __slots__ = ['slots', 'max_slots', 'utility_prefs']

    def __init__(self, max_slots: int, utility_pref_0: float, utility_pref_1: float):
        self.max_slots = int(np.clip(max_slots, 2, 6))
        self.slots = np.zeros((self.max_slots, 2), dtype=np.float32)  # [substance_id, amount]
        self.slots[:, 0] = -1  # empty slots
        # 4-dim utility preference from 2 genome genes
        self.utility_prefs = np.array([
            utility_pref_0,
            1.0 - utility_pref_0,
            utility_pref_1,
            1.0 - utility_pref_1,
        ], dtype=np.float32)

    def add_item(self, substance_id: int, amount: float) -> float:
        """Add item to inventory. Returns overflow amount."""
        # Try existing slot first
        for i in range(self.max_slots):
            if int(self.slots[i, 0]) == substance_id:
                space = 1.0 - self.slots[i, 1]
                added = min(amount, space)
                self.slots[i, 1] += added
                return amount - added
        # Find empty slot
        for i in range(self.max_slots):
            if self.slots[i, 0] < 0:
                self.slots[i, 0] = substance_id
                self.slots[i, 1] = min(amount, 1.0)
                return max(0, amount - 1.0)
        return amount  # no space

    def remove_item(self, slot_idx: int, amount: float) -> float:
        """Remove from slot. Returns actual removed."""
        if slot_idx < 0 or slot_idx >= self.max_slots:
            return 0.0
        if self.slots[slot_idx, 0] < 0:
            return 0.0
        actual = min(amount, self.slots[slot_idx, 1])
        self.slots[slot_idx, 1] -= actual
        if self.slots[slot_idx, 1] < 0.001:
            self.slots[slot_idx, 0] = -1
            self.slots[slot_idx, 1] = 0
        return actual

    def get_utility(self, substance_id: int, amount: float, substance_props) -> float:
        """Compute subjective value. substance_props shape: (n_substances, n_props)."""
        if substance_id < 0 or substance_id >= len(substance_props):
            return 0.0
        props = substance_props[substance_id][:4]
        return float(np.dot(self.utility_prefs, props) * amount)

    def total_utility(self, substance_props) -> float:
        total = 0.0
        for i in range(self.max_slots):
            sid = int(self.slots[i, 0])
            if sid >= 0:
                total += self.get_utility(sid, self.slots[i, 1], substance_props)
        return total

    def best_offer_for(self, other: 'Inventory', substance_props) -> tuple | None:
        """Find a trade that benefits both parties.
        Returns (my_slot, their_slot, my_amount, their_amount) or None."""
        best_mutual = 0.0
        best_trade = None

        for mi in range(self.max_slots):
            msid = int(self.slots[mi, 0])
            if msid < 0 or self.slots[mi, 1] < 0.05:
                continue
            my_util = self.get_utility(msid, 0.1, substance_props)
            their_util_of_mine = other.get_utility(msid, 0.1, substance_props)

            for ti in range(other.max_slots):
                tsid = int(other.slots[ti, 0])
                if tsid < 0 or other.slots[ti, 1] < 0.05:
                    continue
                their_util = other.get_utility(tsid, 0.1, substance_props)
                my_util_of_theirs = self.get_utility(tsid, 0.1, substance_props)

                # Both must gain: I value theirs more than mine, they value mine more than theirs
                my_gain = my_util_of_theirs - my_util
                their_gain = their_util_of_mine - their_util

                if my_gain > 0 and their_gain > 0:
                    mutual = my_gain + their_gain
                    if mutual > best_mutual:
                        best_mutual = mutual
                        best_trade = (mi, ti, 0.1, 0.1)

        return best_trade

    @property
    def fullness(self) -> float:
        used = sum(1 for i in range(self.max_slots) if self.slots[i, 0] >= 0)
        return used / max(1, self.max_slots)

    @property
    def item_count(self) -> int:
        return sum(1 for i in range(self.max_slots) if self.slots[i, 0] >= 0)


class TrustMemory:
    """Tracks trust scores toward other agents based on interaction outcomes."""

    __slots__ = ['_scores', '_last_tick', '_max_entries']

    def __init__(self, max_entries: int = 24):
        self._scores: dict[int, float] = {}
        self._last_tick: dict[int, int] = {}
        self._max_entries = max_entries

    def get_trust(self, other_id: int) -> float:
        return self._scores.get(other_id, 0.5)

    def update_trust(self, other_id: int, outcome: float, tick: int):
        """outcome in [-1, 1]. Positive = good interaction, negative = bad."""
        old = self._scores.get(other_id, 0.5)
        lr = 0.15
        new = old + lr * outcome
        self._scores[other_id] = float(np.clip(new, 0, 1))
        self._last_tick[other_id] = tick
        self._prune()

    def decay(self, tick: int):
        """Trust decays toward 0.5 for stale entries."""
        to_remove = []
        for aid, last in self._last_tick.items():
            if tick - last > 100:
                old = self._scores.get(aid, 0.5)
                self._scores[aid] = old + 0.01 * (0.5 - old)
                if abs(self._scores[aid] - 0.5) < 0.02:
                    to_remove.append(aid)
        for aid in to_remove:
            del self._scores[aid]
            del self._last_tick[aid]

    def _prune(self):
        if len(self._scores) > self._max_entries:
            sorted_ids = sorted(self._last_tick.keys(), key=lambda k: self._last_tick[k])
            while len(self._scores) > self._max_entries:
                aid = sorted_ids.pop(0)
                del self._scores[aid]
                del self._last_tick[aid]

    @property
    def mean_trust(self) -> float:
        if not self._scores:
            return 0.5
        return float(np.mean(list(self._scores.values())))
