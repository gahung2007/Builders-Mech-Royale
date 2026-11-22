"""Survival-first Mech Royale controller for mechfight/1.

The policy favors safe zone positioning, kiting, nearby cubes and upgrades, then
accurate opportunistic fire. Navigation uses a coarse static occupancy grid built
from the read-only arena map.
"""
from __future__ import annotations

import array
import heapq
import json
import math
import os
import random
import sys
from typing import Any

import mechfight as mf

TAU = 2.0 * math.pi
SQRT2 = math.sqrt(2.0)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def wrap_angle(v: float) -> float:
    while v > math.pi:
        v -= TAU
    while v < -math.pi:
        v += TAU
    return v


def unit2(x: float, y: float) -> tuple[float, float]:
    n = math.hypot(x, y)
    return (0.0, 0.0) if n < 1e-8 else (x / n, y / n)


def dist2(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_ring(x: float, y: float, ring: list[float]) -> bool:
    inside = False
    n = len(ring) // 2
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def ring_distance_sq(x: float, y: float, ring: list[float]) -> float:
    best = float("inf")
    n = len(ring) // 2
    for i in range(n):
        ax, ay = ring[2 * i], ring[2 * i + 1]
        bx, by = ring[2 * ((i + 1) % n)], ring[2 * ((i + 1) % n) + 1]
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        t = clamp(((x - ax) * dx + (y - ay) * dy) / den, 0.0, 1.0) if den > 1e-12 else 0.0
        px, py = ax + t * dx, ay + t * dy
        best = min(best, (x - px) ** 2 + (y - py) ** 2)
    return best


class WorldNav:
    """Coarse ground navigation and line checks from the static arena map."""

    def __init__(self, map_dir: str, mech_height: float = 14.0, cell: float = 8.0):
        world_path = os.path.join(map_dir, "v2", "world.json")
        if not os.path.isfile(world_path):
            world_path = os.path.join(map_dir, "world.json")
        with open(world_path, "r", encoding="utf-8") as f:
            world = json.load(f)

        terrain = world["terrain"]
        terrain_path = os.path.join(map_dir, terrain["file"])
        if not os.path.isfile(terrain_path):
            terrain_path = os.path.join(map_dir, "v2", terrain["file"])
        with open(terrain_path, "rb") as f:
            raw = f.read()
        self.terrain = array.array("h")
        self.terrain.frombytes(raw[: len(raw) - (len(raw) % 2)])
        if sys.byteorder != "little":
            self.terrain.byteswap()
        self.terrain_nx = int(terrain["nx"])
        self.terrain_ny = int(terrain["ny"])
        self.terrain_min_x = float(terrain["minX"])
        self.terrain_min_y = float(terrain["minY"])
        self.terrain_step = float(terrain.get("step", 1.0))
        self.terrain_scale = float(terrain.get("scale", 0.01))

        bounds = world["bounds"]
        self.min_x = self.terrain_min_x
        self.min_y = self.terrain_min_y
        self.cell = cell
        self.nx = max(1, math.ceil((float(bounds["maxX"]) - self.min_x) / cell))
        self.ny = max(1, math.ceil((float(bounds["maxY"]) - self.min_y) / cell))
        self.blocked = bytearray(self.nx * self.ny)
        self.ground = [0.0] * (self.nx * self.ny)
        for j in range(self.ny):
            y = self.min_y + (j + 0.5) * cell
            for i in range(self.nx):
                x = self.min_x + (i + 0.5) * cell
                self.ground[j * self.nx + i] = self.ground_z(x, y)

        buffer_sq = (2.8 + cell * 0.08) ** 2
        for prism in world.get("prisms", []):
            if prism.get("kind") not in ("building", "stadium", "bridge"):
                continue
            ring = prism.get("ring", [])
            if len(ring) < 6:
                continue
            xs, ys = ring[0::2], ring[1::2]
            i0 = max(0, int(math.floor((min(xs) - self.min_x) / cell)) - 1)
            i1 = min(self.nx - 1, int(math.ceil((max(xs) - self.min_x) / cell)) + 1)
            j0 = max(0, int(math.floor((min(ys) - self.min_y) / cell)) - 1)
            j1 = min(self.ny - 1, int(math.ceil((max(ys) - self.min_y) / cell)) + 1)
            base = float(prism.get("base", -1e6))
            top = float(prism.get("top", 1e6))
            holes = prism.get("holes", [])
            for j in range(j0, j1 + 1):
                y = self.min_y + (j + 0.5) * cell
                row = j * self.nx
                for i in range(i0, i1 + 1):
                    idx = row + i
                    z = self.ground[idx]
                    if top < z + 0.2 or base > z + mech_height:
                        continue
                    x = self.min_x + (i + 0.5) * cell
                    inside = point_in_ring(x, y, ring)
                    near_edge = ring_distance_sq(x, y, ring) <= buffer_sq
                    if not inside and not near_edge:
                        continue
                    in_hole = any(point_in_ring(x, y, h) for h in holes if len(h) >= 6)
                    near_hole = any(ring_distance_sq(x, y, h) <= buffer_sq for h in holes if len(h) >= 6)
                    if (inside and not in_hole) or near_hole:
                        self.blocked[idx] = 1

        self.neighbors = []
        for j in range(self.ny):
            for i in range(self.nx):
                ns = []
                for dx, dy, cost in (
                    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
                ):
                    x, y = i + dx, j + dy
                    if 0 <= x < self.nx and 0 <= y < self.ny:
                        ns.append((y * self.nx + x, cost, dx != 0 and dy != 0))
                self.neighbors.append(ns)
        self._path: list[int] = []
        self._path_goal = -1
        self._path_cursor = 0

    def ground_z(self, x: float, y: float) -> float:
        i = int(clamp(round((x - self.terrain_min_x) / self.terrain_step), 0, self.terrain_nx - 1))
        j = int(clamp(round((y - self.terrain_min_y) / self.terrain_step), 0, self.terrain_ny - 1))
        k = j * self.terrain_nx + i
        return self.terrain[k] * self.terrain_scale if 0 <= k < len(self.terrain) else 0.0

    def cell_for(self, x: float, y: float) -> int:
        i = int(clamp((x - self.min_x) / self.cell, 0, self.nx - 1))
        j = int(clamp((y - self.min_y) / self.cell, 0, self.ny - 1))
        return j * self.nx + i

    def center(self, idx: int) -> tuple[float, float]:
        j, i = divmod(idx, self.nx)
        return self.min_x + (i + 0.5) * self.cell, self.min_y + (j + 0.5) * self.cell

    def _nearest_free(self, idx: int) -> int:
        if not self.blocked[idx]:
            return idx
        j0, i0 = divmod(idx, self.nx)
        best, best_d = idx, float("inf")
        for r in range(1, 10):
            found = False
            for j in range(max(0, j0 - r), min(self.ny, j0 + r + 1)):
                for i in range(max(0, i0 - r), min(self.nx, i0 + r + 1)):
                    if abs(i - i0) != r and abs(j - j0) != r:
                        continue
                    cand = j * self.nx + i
                    if not self.blocked[cand]:
                        d = (i - i0) ** 2 + (j - j0) ** 2
                        if d < best_d:
                            best, best_d, found = cand, d, True
            if found:
                return best
        return best

    def clear_line(self, a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> bool:
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        steps = max(1, int(d / (self.cell * 0.38)))
        for k in range(1, steps):
            t = k / steps
            if self.blocked[self.cell_for(a[0] + dx * t, a[1] + dy * t)]:
                return False
        return True

    def _astar(self, start: int, goal: int) -> list[int]:
        if start == goal:
            return [start]
        start, goal = self._nearest_free(start), self._nearest_free(goal)
        costs = {start: 0.0}
        parent: dict[int, int] = {}
        gx, gy = goal % self.nx, goal // self.nx
        heap = [(0.0, 0.0, start)]
        while heap:
            _, g, here = heapq.heappop(heap)
            if g != costs.get(here):
                continue
            if here == goal:
                break
            hi, hj = here % self.nx, here // self.nx
            for nxt, step_cost, diagonal in self.neighbors[here]:
                if self.blocked[nxt]:
                    continue
                if diagonal:
                    ni, nj = nxt % self.nx, nxt // self.nx
                    if self.blocked[hj * self.nx + ni] or self.blocked[nj * self.nx + hi]:
                        continue
                ng = g + step_cost
                if ng >= costs.get(nxt, float("inf")):
                    continue
                costs[nxt] = ng
                parent[nxt] = here
                heuristic = math.hypot(gx - nxt % self.nx, gy - nxt // self.nx)
                heapq.heappush(heap, (ng + heuristic, ng, nxt))
        if goal not in costs:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def waypoint(self, start_xy: list[float], goal_xy: list[float]) -> tuple[float, float]:
        if self.clear_line(start_xy, goal_xy):
            self._path = []
            self._path_goal = -1
            return goal_xy[0], goal_xy[1]
        raw_goal = self.cell_for(goal_xy[0], goal_xy[1])
        goal = self._nearest_free(raw_goal)
        final_goal = goal_xy if goal == raw_goal else self.center(goal)
        start = self._nearest_free(self.cell_for(start_xy[0], start_xy[1]))
        if not self._path or goal != self._path_goal:
            self._path = self._astar(start, goal)
            self._path_goal, self._path_cursor = goal, 0
        if not self._path:
            return final_goal
        end = min(len(self._path), self._path_cursor + 7)
        nearest_i = min(
            range(self._path_cursor, end),
            key=lambda q: (self.center(self._path[q])[0] - start_xy[0]) ** 2
                         + (self.center(self._path[q])[1] - start_xy[1]) ** 2,
        )
        if dist2(start_xy, self.center(self._path[nearest_i])) > self.cell * 1.7:
            self._path = self._astar(start, goal)
            self._path_cursor = 0
        else:
            self._path_cursor = nearest_i
        while self._path_cursor < len(self._path) - 1:
            if dist2(start_xy, self.center(self._path[self._path_cursor + 1])) > self.cell * 0.75:
                break
            self._path_cursor += 1
        chosen = self.center(self._path[min(self._path_cursor + 1, len(self._path) - 1)])
        for q in range(self._path_cursor + 2, min(len(self._path), self._path_cursor + 7)):
            candidate = self.center(self._path[q])
            if self.clear_line(start_xy, candidate):
                chosen = candidate
        return final_goal if self._path_cursor >= len(self._path) - 2 else chosen


class Bot:
    name = "Oval Warden"

    def __init__(self, hello: mf.Hello):
        self.config = hello.get("config", {})
        self.rng = random.Random(int(hello.get("seed", 0)))
        self.patrol_goal: list[float] | None = None
        self.patrol_until = -1.0
        self.last_jump_tick = -1000
        self.nav: WorldNav | None = None
        map_dir = str(hello.get("map", {}).get("dir", "/arena"))
        if not os.path.isabs(map_dir):
            local = os.path.join(os.path.dirname(os.path.abspath(__file__)), map_dir)
            if os.path.isdir(local):
                map_dir = local
        try:
            self.nav = WorldNav(map_dir)
        except Exception as exc:
            print(f"map navigation unavailable: {exc}", file=sys.stderr)

    def _upgrade_choice(self, offer: dict[str, Any] | None, weapons: list[Any]) -> int | None:
        if not offer:
            return None
        choices = offer.get("choices", [])
        if not choices:
            return 0
        weights = {
            "armor": 100, "plating": 95, "servos": 92, "nanites": 89,
            "sensors": 82, "jets": 78, "gyros": 72, "capacitor": 48, "reactor": 50,
            "gun_damage": 80, "gun_rate": 79, "gun_velocity": 52, "gun_mag": 55,
            "mortar_reload": 74, "mortar_velocity": 55, "mortar_splash": 48,
            "laser_charge": 84, "laser_damage": 87, "laser_pierce": 60,
            "lightning_range": 76, "lightning_efficiency": 79, "lightning_chain": 73,
        }
        held = {w.get("kind") for w in weapons if isinstance(w, dict)}
        def score(choice: dict[str, Any]) -> float:
            ident = choice.get("id", "")
            value = float(weights.get(ident, 0))
            for prefix, kind in (("mortar_", "mortar"), ("laser_", "laser"), ("lightning_", "lightning")):
                if ident.startswith(prefix) and kind not in held:
                    value -= 40
            return value
        return max(range(len(choices)), key=lambda i: score(choices[i]))

    def _inside_circle(self, point: list[float] | tuple[float, float],
                       center: list[float] | tuple[float, float], radius: float,
                       margin: float = 0.0) -> bool:
        return dist2(point, center) <= max(0.0, radius - margin)

    def _circle_entry(self, pos: list[float], center: list[float], radius: float,
                      margin: float = 25.0) -> list[float]:
        safe_r = max(0.0, radius - margin)
        dx, dy = pos[0] - center[0], pos[1] - center[1]
        d = math.hypot(dx, dy)
        if d <= safe_r:
            return [pos[0], pos[1]]
        if d < 1e-6:
            return [center[0], center[1]]
        return [center[0] + dx / d * safe_r, center[1] + dy / d * safe_r]

    def _zone_goal(self, me: dict[str, Any], zone: dict[str, Any], t: float) -> tuple[list[float] | None, bool]:
        pos = me["pos"]
        center = zone.get("center", [0.0, 0.0])
        radius = float(zone.get("radius", 0.0))
        next_center = zone.get("next_center", center)
        next_radius = float(zone.get("next_radius", radius))
        start = float(zone.get("shrink_start", t + 9999.0))
        if not bool(me.get("in_zone", True)):
            return self._circle_entry(pos, center, radius, 12.0), True
        next_safe = max(0.0, next_radius - max(18.0, min(45.0, next_radius * 0.12)))
        future_distance = dist2(pos, next_center)
        if future_distance > next_safe:
            travel = future_distance - next_safe
            time_left = max(0.0, start - t)
            if t >= start or time_left <= max(8.0, travel / 10.5 + 9.0):
                return self._circle_entry(pos, next_center, next_radius, 22.0), True
        if not self._inside_circle(pos, center, radius, 18.0):
            return self._circle_entry(pos, center, radius, 12.0), True
        return None, False

    def _safe_loot_point(self, xy: list[float], zone: dict[str, Any], t: float) -> bool:
        center = zone.get("center", [0.0, 0.0])
        radius = float(zone.get("radius", 0.0))
        if not self._inside_circle(xy, center, radius, 24.0):
            return False
        if float(zone.get("shrink_start", t + 9999.0)) - t < 18.0:
            if not self._inside_circle(xy, zone.get("next_center", center),
                                       float(zone.get("next_radius", radius)), 20.0):
                return False
        return True

    def _patrol(self, pos: list[float], zone: dict[str, Any], t: float,
                enemies: list[dict[str, Any]], cubes: list[dict[str, Any]]) -> list[float]:
        if self.patrol_goal is not None and t < self.patrol_until and dist2(pos, self.patrol_goal) > 12.0:
            return self.patrol_goal
        center = zone.get("center", [0.0, 0.0])
        radius = float(zone.get("radius", 0.0))
        next_center = zone.get("next_center", center)
        next_radius = float(zone.get("next_radius", radius))
        start = float(zone.get("shrink_start", t + 9999.0))
        phase = self.rng.random() * TAU
        candidates = []
        for k in range(24):
            angle = phase + k * TAU / 24.0
            travel = 55.0 + (k % 3) * 17.0
            point = [pos[0] + math.cos(angle) * travel, pos[1] + math.sin(angle) * travel]
            if not self._inside_circle(point, center, radius, 20.0):
                point = self._circle_entry(point, center, radius, 25.0)
            if start - t < 20.0 and not self._inside_circle(point, next_center, next_radius, 25.0):
                continue
            nearest = min((dist2(point, e["pos"]) for e in enemies), default=220.0)
            score = min(220.0, nearest) - 0.08 * travel + self.rng.random() * 3.0
            for cube in cubes:
                cd = dist2(point, cube["pos"])
                if cd < 30.0:
                    score += (30.0 - cd) * 0.55
            candidates.append((score, point))
        self.patrol_goal = max(candidates, key=lambda x: x[0])[1] if candidates else pos[:2]
        self.patrol_until = t + 7.0 + self.rng.random() * 6.0
        return self.patrol_goal

    def _weapon_rating(self, kind: str, distance: float) -> float:
        if kind == "laser":
            return 3.6 if distance > 45.0 else 2.6
        if kind == "mortar":
            return 2.8 if distance > 35.0 else 1.8
        if kind == "lightning":
            return 3.0 if distance < 55.0 else 1.6
        return 0.0

    def _weapon_goal(self, me: dict[str, Any], obs: dict[str, Any], t: float,
                     enemies: list[dict[str, Any]], slots: list[Any]) -> tuple[list[float] | None, dict[str, Any] | None]:
        carried = {w.get("kind") for w in slots if isinstance(w, dict)}
        slot1 = slots[1]
        ground = self.nav.ground_z if self.nav else lambda x, y: me["pos"][2]
        nearest_enemy = min((dist2(me["pos"], e["pos"]) for e in enemies), default=999.0)
        best, best_score = None, float("inf")
        for pickup in obs.get("pickups", []):
            kind = pickup.get("kind", "")
            p = pickup.get("pos", [0.0, 0.0, 0.0])
            d = dist2(me["pos"], p)
            if kind == "gun" or kind in carried or d > 62.0 or nearest_enemy < 58.0:
                continue
            if abs(float(p[2]) - ground(p[0], p[1])) > 2.8 or not self._safe_loot_point(p, obs["zone"], t):
                continue
            if slot1 is not None:
                old = self._weapon_rating(slot1.get("kind", ""), nearest_enemy)
                new = self._weapon_rating(kind, nearest_enemy)
                if new < old + 0.35:
                    continue
            score = d - self._weapon_rating(kind, nearest_enemy) * 8.0
            if score < best_score:
                best, best_score = pickup, score
        return (best["pos"][:2], best) if best else (None, None)

    def _supply_goal(self, me: dict[str, Any], obs: dict[str, Any], t: float,
                     enemies: list[dict[str, Any]]) -> tuple[list[float] | None, bool]:
        me_id = me.get("id")
        pos = me["pos"]
        closest = min((dist2(pos, e["pos"]) for e in enemies), default=999.0)
        for drop in obs.get("drops", []):
            if float(drop.get("lands_in", 999.0)) > 0.0:
                continue
            holder = drop.get("holder")
            if holder is not None and holder != me_id:
                continue
            p = drop.get("pos", [0.0, 0.0, 0.0])
            d = dist2(pos, p)
            if d > 155.0 or closest < 90.0 or not self._safe_loot_point(p, obs["zone"], t):
                continue
            if d <= 6.5:
                return pos[:2], True
            return p[:2], False
        return None, False

    def _target(self, me: dict[str, Any], obs: dict[str, Any],
                mechs: list[dict[str, Any]], npcs: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = mechs
        if not candidates:
            # Drones can be a worthwhile cube source if they are already nearby and isolated.
            candidates = [n for n in npcs if dist2(me["pos"], n["pos"]) < 82.0]
        if not candidates:
            return None
        attackers = {e.get("by") for e in obs.get("events", [])
                     if e.get("type") == "damaged" and e.get("by") is not None}
        def score(actor: dict[str, Any]) -> float:
            d = math.dist(me["pos"], actor["pos"])
            hp = float(actor.get("hp", actor.get("max_hp", 1000.0)))
            maximum = max(1.0, float(actor.get("max_hp", hp)))
            return d + 65.0 * hp / maximum - (75.0 if actor.get("id") in attackers else 0.0)
        return min(candidates, key=score)

    def _lead(self, me: dict[str, Any], target: dict[str, Any], speed: float | None,
              lead_time: float | None = None) -> tuple[float, float, float]:
        pos, vel = me["pos"], target.get("vel", [0.0, 0.0, 0.0])
        scale = 1.0 + 0.05 * float(target.get("cubes", 0.0))
        muzzle = float(me.get("stats", {}).get("muzzle_height", 7.0))
        r = [target["pos"][0] - pos[0], target["pos"][1] - pos[1],
             target["pos"][2] + 4.5 * scale - (pos[2] + muzzle)]
        if lead_time is not None:
            tm = max(0.0, lead_time)
        elif speed and speed > 1.0:
            tm = math.sqrt(sum(v * v for v in r)) / speed
            for _ in range(3):
                future = [r[i] + vel[i] * tm for i in range(3)]
                tm = math.sqrt(sum(v * v for v in future)) / speed
        else:
            tm = 0.0
        rx, ry, rz = r[0] + vel[0] * tm, r[1] + vel[1] * tm, r[2] + vel[2] * tm
        horizontal = max(1e-6, math.hypot(rx, ry))
        return math.atan2(ry, rx), math.atan2(rz, horizontal), math.sqrt(rx * rx + ry * ry + rz * rz)

    def _mortar_aim(self, me: dict[str, Any], target: dict[str, Any],
                    speed: float, gravity: float) -> tuple[float, float] | None:
        horizontal = dist2(me["pos"], target["pos"])
        if horizontal < 1.0 or speed <= 1.0 or gravity <= 0.1:
            return None
        muzzle_z = me["pos"][2] + float(me.get("stats", {}).get("muzzle_height", 7.0)) + 2.5
        target_z = target["pos"][2] + 4.5 * (1.0 + 0.05 * float(target.get("cubes", 0.0)))
        v2 = speed * speed
        disc = v2 * v2 - gravity * (gravity * horizontal * horizontal + 2.0 * (target_z - muzzle_z) * v2)
        if disc < 0.0:
            return None
        pitch = math.atan((v2 - math.sqrt(disc)) / (gravity * horizontal))
        flight = horizontal / max(1.0, speed * math.cos(pitch))
        if flight > 5.5:
            return None
        future = dict(target)
        future["pos"] = [target["pos"][i] + target.get("vel", [0.0, 0.0, 0.0])[i] * flight for i in range(3)]
        yaw, _, _ = self._lead(me, future, None)
        return yaw, pitch

    def _projectile_dodge(self, me: dict[str, Any], obs: dict[str, Any]) -> tuple[tuple[float, float], float, bool]:
        pos, vel = me["pos"], me.get("vel", [0.0, 0.0, 0.0])
        best_away, best_risk, jump = (0.0, 0.0), 0.0, False
        gravity = float(self.config.get("weapons", {}).get("mortar", {}).get("gravity", 22.0))
        for projectile in obs.get("projectiles", []):
            p, v = projectile.get("pos", [0.0, 0.0, 0.0]), projectile.get("vel", [0.0, 0.0, 0.0])
            if projectile.get("kind") == "mortar":
                ground = self.nav.ground_z(p[0], p[1]) if self.nav else pos[2]
                dz = max(0.0, p[2] - ground)
                impact_t = (v[2] + math.sqrt(max(0.0, v[2] * v[2] + 2.0 * gravity * dz))) / gravity
                if impact_t <= 0.0 or impact_t > 3.5:
                    continue
                ix, iy = p[0] + v[0] * impact_t, p[1] + v[1] * impact_t
                mx, my = pos[0] + vel[0] * impact_t, pos[1] + vel[1] * impact_t
                d = math.hypot(mx - ix, my - iy)
                if d < 16.0:
                    risk = (16.0 - d) / 16.0
                    away = unit2(mx - ix, my - iy)
                    if away == (0.0, 0.0):
                        away = unit2(-v[0], -v[1])
                    if risk > best_risk:
                        best_away, best_risk, jump = away, risk, impact_t < 0.7
            else:
                rx, ry, rz = p[0] - pos[0], p[1] - pos[1], p[2] - (pos[2] + 4.5)
                rvx, rvy, rvz = v[0] - vel[0], v[1] - vel[1], v[2] - vel[2]
                vv = rvx * rvx + rvy * rvy + rvz * rvz
                closest_t = clamp(-(rx * rvx + ry * rvy + rz * rvz) / vv, 0.0, 0.65) if vv > 1e-6 else 0.0
                cx, cy, cz = rx + rvx * closest_t, ry + rvy * closest_t, rz + rvz * closest_t
                d = math.sqrt(cx * cx + cy * cy + cz * cz)
                if d < 5.5:
                    risk = (5.5 - d) / 5.5
                    away = unit2(-cx, -cy)
                    if away == (0.0, 0.0):
                        away = unit2(-v[1], v[0])
                    if risk > best_risk:
                        best_away, best_risk = away, risk
        return best_away, best_risk, jump

    def _movement_threat(self, me: dict[str, Any], obs: dict[str, Any],
                         actors: list[dict[str, Any]]) -> tuple[tuple[float, float], float, float]:
        pos = me["pos"]
        ax = ay = 0.0
        nearest = 999.0
        for actor in actors:
            d = dist2(pos, actor["pos"])
            nearest = min(nearest, d)
            limit = 100.0 if actor.get("kind") == "npc" else 155.0
            if d < limit:
                away = unit2(pos[0] - actor["pos"][0], pos[1] - actor["pos"][1])
                weight = (limit - d) / limit
                if actor.get("kind") == "npc":
                    weight *= 0.45
                ax += away[0] * weight
                ay += away[1] * weight
        for sound in obs.get("sounds", []):
            if float(sound.get("age", 0.0)) > 0.8:
                continue
            kind, dist_class = sound.get("kind", ""), sound.get("dist", "far")
            weight = 0.9 if dist_class == "near" else 0.35 if dist_class == "mid" else 0.10
            if kind in ("step", "boost", "jet", "death", "crate", "drop"):
                weight *= 0.4
            bearing = float(sound.get("bearing", 0.0))
            ax -= math.cos(bearing) * weight
            ay -= math.sin(bearing) * weight
        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                bearing = float(event["bearing"])
                weight = 1.6 if event.get("rear") else 1.0
                ax -= math.cos(bearing) * weight
                ay -= math.sin(bearing) * weight
        return unit2(ax, ay), nearest, math.hypot(ax, ay)

    def _attack(self, me: dict[str, Any], target: dict[str, Any] | None, slots: list[Any],
                actions: mf.Action, nearest_enemy: float) -> list[float]:
        current_aim = me.get("aim", [0.0, 0.0])
        gun_slot = slots[0] if isinstance(slots[0], dict) else {}
        if gun_slot.get("kind") == "gun" and int(gun_slot.get("ammo", 0)) <= 3 \
                and float(gun_slot.get("reload", 0.0)) <= 0.0 and nearest_enemy > 72.0:
            actions["reload"] = True
        if target is None:
            recent = [s for s in getattr(self, "_sounds", [])
                      if float(s.get("age", 1.0)) < 0.7 and s.get("kind") in
                      ("gun", "mortar", "explosion", "laser", "laser_charge", "lightning", "npc_gun")]
            if recent:
                sound = min(recent, key=lambda s: (0 if s.get("dist") == "near" else 1, float(s.get("age", 0.0))))
                return [float(sound.get("bearing", current_aim[0])), 0.0]
            move = actions.get("move", [0.0, 0.0])
            return [math.atan2(move[1], move[0]), 0.0] if math.hypot(*move) > 0.1 else list(current_aim)

        gun = slots[0] if isinstance(slots[0], dict) else {}
        gun_stats = gun.get("stats", self.config.get("weapons", {}).get("gun", {}))
        gun_yaw, gun_pitch, gun_range = self._lead(me, target, float(gun_stats.get("speed", 252.0)))
        desired = [gun_yaw, gun_pitch]
        mode = "gun"
        special = slots[1] if isinstance(slots[1], dict) else None
        distance = math.dist(me["pos"], target["pos"])
        shot_clear = not self.nav or self.nav.clear_line(me["pos"][:2], target["pos"][:2])
        special_stats: dict[str, Any] = {}
        charge = 0.0
        mortar_aim = None

        if special and shot_clear:
            kind = special.get("kind")
            special_stats = special.get("stats", self.config.get("weapons", {}).get(kind, {}))
            cooldown = float(special.get("cooldown", 0.0))
            if kind == "lightning" and distance <= float(special_stats.get("range", 45.0)):
                mode = "lightning"
            elif kind == "laser":
                charge = float(special.get("charge", 0.0))
                limit = float(special_stats.get("range", 280.0))
                if charge > 0.0 or (45.0 < distance <= limit and cooldown <= 0.0):
                    mode = "laser"
                    remaining = max(0.04, float(special_stats.get("charge", 0.75)) - charge)
                    ly, lp, _ = self._lead(me, target, None, remaining)
                    desired = [ly, lp]
            elif kind == "mortar" and 48.0 < distance <= 225.0 and cooldown <= 0.0:
                mortar_aim = self._mortar_aim(
                    me, target, float(special_stats.get("speed", 72.0)),
                    float(special_stats.get("gravity", 22.0)),
                )
                if mortar_aim is not None:
                    mode = "mortar"
                    desired = [mortar_aim[0], mortar_aim[1]]

        desired[0] = wrap_angle(desired[0])
        desired[1] = clamp(desired[1], float(self.config.get("mech", {}).get("min_pitch", -0.7)),
                           float(self.config.get("mech", {}).get("max_pitch", 1.3)))
        actions["aim"] = desired
        actual = me.get("aim", desired)
        yaw_error = abs(wrap_angle(desired[0] - float(actual[0])))
        pitch_error = abs(desired[1] - float(actual[1]))
        clear = shot_clear and (target.get("kind") != "npc" or distance <= 85.0)

        if mode == "gun" or mode == "lightning":
            yaw_tol = min(0.08, 2.8 / max(10.0, gun_range))
            pitch_tol = min(0.07, 4.2 / max(10.0, gun_range))
            if clear and gun.get("kind") == "gun" and gun_range <= float(gun_stats.get("range", 230.0)) \
                    and yaw_error <= yaw_tol and pitch_error <= pitch_tol \
                    and int(gun.get("ammo", 0)) > 0 and float(gun.get("reload", 0.0)) <= 0.0:
                actions["fire"][0] = True
            if mode == "lightning" and clear:
                cone = math.radians(float(special_stats.get("cone_deg", 30.0)) * 0.5)
                if distance <= float(special_stats.get("range", 45.0)) and yaw_error <= cone * 0.7 \
                        and float(me.get("energy", 0.0)) >= float(special_stats.get("energy", 3.0)):
                    actions["fire"][1] = True
        elif mode == "laser" and clear:
            charge_limit = float(special_stats.get("charge", 0.75))
            if charge > 0.0:
                actions["fire"][1] = charge < charge_limit - 0.04
            elif yaw_error < 0.10 and pitch_error < 0.10:
                actions["fire"][1] = True
        elif mode == "mortar" and clear and mortar_aim is not None:
            if yaw_error < 0.05 and pitch_error < 0.05:
                actions["fire"][1] = True

        if gun.get("kind") == "gun" and int(gun.get("ammo", 0)) <= 3 \
                and float(gun.get("reload", 0.0)) <= 0.0 and nearest_enemy > 72.0:
            actions["reload"] = True
        return desired

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        pos = me["pos"]
        t = float(obs.get("t", 0.0))
        tick = int(obs.get("tick", 0))
        zone = obs.get("zone", {})
        slots = list(me.get("weapons", [None, None]))
        while len(slots) < 2:
            slots.append(None)
        actors = list(obs.get("mechs", []))
        mechs = [m for m in actors if m.get("kind") == "mech"]
        npcs = [m for m in actors if m.get("kind") == "npc"]
        target = self._target(me, obs, mechs, npcs)
        self._sounds = obs.get("sounds", [])
        actions: mf.Action = {"tick": tick, "move": [0.0, 0.0], "boost": False,
                              "aim": me.get("aim", [0.0, 0.0]), "fire": [False, False]}

        choice = self._upgrade_choice(obs.get("offer"), slots)
        if choice is not None:
            actions["pick"] = choice

        goal, holding_drop = self._supply_goal(me, obs, t, actors)
        urgent_goal, urgent_zone = self._zone_goal(me, zone, t)
        if urgent_goal is not None:
            goal, holding_drop = urgent_goal, False

        nearest_enemy = min((dist2(pos, e["pos"]) for e in actors), default=999.0)
        if goal is None and float(me.get("cubes", 0.0)) < 12.0 and nearest_enemy >= 68.0:
            limit = 70.0 if float(me.get("cubes", 0.0)) < 10.0 else 24.0
            best, best_d = None, float("inf")
            for cube in obs.get("cubes", []):
                p = cube.get("pos", [0.0, 0.0, 0.0])
                d = dist2(pos, p)
                if d > limit or not self._safe_loot_point(p, zone, t):
                    continue
                if self.nav and abs(p[2] - self.nav.ground_z(p[0], p[1])) > 2.8:
                    continue
                if d < best_d:
                    best, best_d = cube, d
            if best:
                goal = best["pos"][:2]

        weapon_pick = None
        if goal is None and nearest_enemy >= 58.0:
            goal, weapon_pick = self._weapon_goal(me, obs, t, actors, slots)
        if goal is None:
            goal = self._patrol(pos, zone, t, actors, obs.get("cubes", []))

        if weapon_pick:
            p = weapon_pick["pos"]
            if dist2(pos, p) <= 4.0 and abs(pos[2] - p[2]) <= 4.0:
                actions["take"] = weapon_pick["id"]
                if slots[1] is not None:
                    actions["drop"] = 1

        move_goal = pos[:2] if holding_drop and dist2(pos, goal) <= 7.0 else goal[:2]
        route_goal = self.nav.waypoint(pos[:2], move_goal) if self.nav else tuple(move_goal)
        move_vec = unit2(route_goal[0] - pos[0], route_goal[1] - pos[1])
        away, nearest_visible, strength = self._movement_threat(me, obs, actors)
        shell_away, shell_risk, should_jump = self._projectile_dodge(me, obs)
        panic = nearest_visible < 46.0 or shell_risk > 0.55
        if shell_risk > 0.0:
            weight = 0.9 + 1.8 * shell_risk
            move_vec = unit2(move_vec[0] + shell_away[0] * weight,
                             move_vec[1] + shell_away[1] * weight)
        if strength > 0.05:
            weight = (1.65 if panic else 0.95) * min(1.4, strength)
            move_vec = unit2(move_vec[0] + away[0] * weight,
                             move_vec[1] + away[1] * weight)
        actions["move"] = [move_vec[0], move_vec[1]]

        energy = float(me.get("energy", 0.0))
        distance_to_goal = dist2(pos, move_goal)
        outside = not bool(me.get("in_zone", True))
        shrinking_soon = float(zone.get("shrink_start", t + 999.0)) - t < 12.0
        if energy > 10.0 and (panic or shell_risk > 0.25 or outside or (shrinking_soon and urgent_zone)):
            actions["boost"] = True
        elif energy > 64.0 and distance_to_goal > 65.0 and tick % 20 < 7:
            actions["boost"] = True
        if should_jump and bool(me.get("grounded", False)) and energy > 35.0 and tick - self.last_jump_tick >= 8:
            actions["jump"] = True
            self.last_jump_tick = tick

        self._attack(me, target, slots, actions, nearest_visible)

        if holding_drop and nearest_visible > 75.0 and shell_risk < 0.2:
            actions["move"] = [0.0, 0.0]
        return actions


mf.run(Bot)
