
import numpy as np
from enum import IntEnum


class StructureType(IntEnum):
    NONE = 0
    WALL = 1
    STORAGE = 2
    TRAP = 3
    FARM = 4
    MARKER = 5
    NEST = 6
    NUM_TYPES = 7


BUILD_COST = {
    StructureType.WALL: 0.15,
    StructureType.STORAGE: 0.12,
    StructureType.TRAP: 0.10,
    StructureType.FARM: 0.20,
    StructureType.MARKER: 0.05,
    StructureType.NEST: 0.25,
}

DURABILITY = {
    StructureType.WALL: 2000,
    StructureType.STORAGE: 1500,
    StructureType.TRAP: 800,
    StructureType.FARM: 3000,
    StructureType.MARKER: 5000,
    StructureType.NEST: 2500,
}


class Structure:
    __slots__ = ['stype', 'x', 'y', 'builder_lineage', 'age', 'durability',
                 'stored_resource', 'signal_value']

    def __init__(self, stype: int, x: int, y: int, builder_lineage: int):
        self.stype = stype
        self.x = x
        self.y = y
        self.builder_lineage = builder_lineage
        self.age = 0
        self.durability = DURABILITY.get(stype, 1000)
        self.stored_resource = 0.0
        self.signal_value = 0.0


class StructureManager:

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.grid = np.zeros((w, h), dtype=np.int8)
        self.structures: dict[tuple[int, int], Structure] = {}
        self.total_built = 0
        self.total_destroyed = 0

    def build(self, stype: int, x: int, y: int, builder_lineage: int) -> bool:
        x, y = x % self.w, y % self.h
        if self.grid[x, y] != StructureType.NONE:
            return False
        if stype <= 0 or stype >= StructureType.NUM_TYPES:
            return False

        s = Structure(stype, x, y, builder_lineage)
        self.structures[(x, y)] = s
        self.grid[x, y] = stype
        self.total_built += 1
        return True

    def destroy(self, x: int, y: int):
        x, y = x % self.w, y % self.h
        if (x, y) in self.structures:
            del self.structures[(x, y)]
            self.grid[x, y] = StructureType.NONE
            self.total_destroyed += 1

    def get_at(self, x: int, y: int) -> Structure | None:
        return self.structures.get((x % self.w, y % self.h))

    def get_type_at(self, x: int, y: int) -> int:
        return int(self.grid[x % self.w, y % self.h])

    def update(self, tick: int):
        to_remove = []
        for pos, s in self.structures.items():
            s.age += 1
            if s.age >= s.durability:
                to_remove.append(pos)
            if s.stype == StructureType.STORAGE:
                s.stored_resource *= 0.999
        for pos in to_remove:
            self.destroy(*pos)

    def deposit_resource(self, x: int, y: int, amount: float) -> float:
        s = self.get_at(x, y)
        if s and s.stype == StructureType.STORAGE:
            space = 1.0 - s.stored_resource
            deposited = min(amount, space)
            s.stored_resource += deposited
            return deposited
        return 0.0

    def withdraw_resource(self, x: int, y: int, amount: float) -> float:
        s = self.get_at(x, y)
        if s and s.stype == StructureType.STORAGE:
            withdrawn = min(amount, s.stored_resource)
            s.stored_resource -= withdrawn
            return withdrawn
        return 0.0

    def is_blocked(self, x: int, y: int) -> bool:
        return self.get_type_at(x, y) == StructureType.WALL

    def get_trap_damage(self, x: int, y: int, agent_lineage: int) -> float:
        s = self.get_at(x, y)
        if s and s.stype == StructureType.TRAP and s.builder_lineage != agent_lineage:
            return 0.15
        return 0.0

    def get_farm_bonus(self, x: int, y: int) -> float:
        s = self.get_at(x, y)
        if s and s.stype == StructureType.FARM:
            return 0.3
        return 0.0

    def get_nest_bonus(self, x: int, y: int) -> bool:
        s = self.get_at(x, y)
        return s is not None and s.stype == StructureType.NEST

    def get_structure_sensor(self, x: int, y: int) -> np.ndarray:
        stype = self.get_type_at(x, y)
        vec = np.zeros(int(StructureType.NUM_TYPES) + 1)
        vec[stype] = 1.0
        s = self.get_at(x, y)
        if s:
            vec[-1] = s.stored_resource
        return vec

    @property
    def stats(self) -> dict:
        counts = {}
        for s in self.structures.values():
            name = StructureType(s.stype).name
            counts[name] = counts.get(name, 0) + 1
        return {
            "total_structures": len(self.structures),
            "total_built_ever": self.total_built,
            **counts,
        }
