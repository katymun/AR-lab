import random
from collections import defaultdict, deque
import json

def load_grammar(path):
    """
    Load grammar from JSON:
      {
        "axiom": "S",
        "rules": {
          "S": [{"p":1.0,"exp":"P"}],
          ...
        }
      }
    Returns (axiom, rules_dict) where rules_dict maps symbol -> [(p, "expansion"), ...]
    Probabilities are normalized automatically.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    axiom = data.get("axiom", "S")
    raw_rules = data["rules"]

    rules = {}
    for sym, choices in raw_rules.items():
        total = sum(c["p"] for c in choices)
        # avoid divide-by-zero; if total==0, treat as uniform
        if total <= 0:
            n = max(1, len(choices))
            probs = [1.0 / n] * n
        else:
            probs = [c["p"] / total for c in choices]

        rules[sym] = list(zip(probs, [c["exp"] for c in choices]))
    return axiom, rules
# ---------------- L-system core ----------------
class LSystem:
    def __init__(self, axiom, rules):
        self.s = axiom
        # rules: dict[str, list[(prob, expansion_str)]]
        self.rules = rules

    def _expand_one(self, ch):
        if ch not in self.rules:
            return ch
        bucket = self.rules[ch]  # [(p, exp_or_fn), ...]
        r = random.random()
        acc = 0.0
        choice = bucket[-1][1]  # default fallback
        for p, exp in bucket:
            acc += p
            if r <= acc:
                choice = exp
                break
        return choice() if callable(choice) else choice
    def iterate(self, n):
        for _ in range(n):
            out = []
            for ch in self.s:
                if ch in ['+','-','[',']',';','.','{','}','U','D']:
                    out.append(ch)
                else:
                    out.append(self._expand_one(ch))
            self.s = ''.join(out)
        return self.s

def ruleset3d():
    # Growth-safe rules: no early termination, no corridor collapse
    return {
        'S': [(1.0, lambda: 'P')],

        # Always continue the spine; no CT. endings here
        'P': [
            (0.60, lambda: 'CP'),        # extend forward
            (0.20, lambda: 'CJP'),       # junction
            (0.10, lambda: 'CVP'),       # vertical event
            (0.10, lambda: 'CRP'),       # small room on spine
        ],

        # Junctions always feed back into P
        'J': [
            (0.55, lambda: '[+CP][-CP]P'),
            (0.25, lambda: '[+CP]P'),
            (0.10, lambda: '[-CP]P'),
            (0.10, lambda: '[+CR][-CR]P'),
        ],

        # Vertical: up/down choices but still return to P
        'V': [
            (0.45, lambda: 'UCP'),
            (0.45, lambda: 'DCP'),
            (0.10, lambda: '[+UCP][-DCP]P'),
        ],

        # Rooms can grow a bit or just mark a spot; they do NOT end the plan
        'R': [
            (0.60, lambda: 'CR'),
            (0.40, lambda: ';'),
        ],

        # Do NOT place T/K/B during rewriting; post-pass only
        'T': [(1.0, lambda: ';')],
        'K': [(1.0, lambda: ';')],
        'B': [(1.0, lambda: ';')],

        # Critical: don't collapse corridors while rewriting
        'C': [(1.0, lambda: 'C')],
    }

# Optional: stream pass to clean plan (bounds & shafts)
def sanitize_plan(plan, zmin=0, zmax=4, max_run_stairs=2):
    """Remove illegal stairs at bounds (heuristic) and cap U/D runs."""
    # Since we don't know z during rewriting, we only collapse long runs.
    out = []
    run_ch, run_len = None, 0
    for ch in plan:
        if ch in ('U','D'):
            if ch == run_ch:
                run_len += 1
            else:
                run_ch, run_len = ch, 1
            if run_len <= max_run_stairs:
                out.append(ch)
            # else: drop extra stairs in the run
        else:
            run_ch, run_len = None, 0
            out.append(ch)
    return ''.join(out)

# ---------------- 3D Turtle / Carver ----------------
FLOOR, WALL = 1, 0

class Dungeon3D:
    def __init__(self, w=60, h=40, zmax=4, start=(30,20,0)):
        self.w, self.h, self.zmax = w, h, zmax
        self.grid = [[[WALL]*w for _ in range(h)] for __ in range(zmax+1)]
        self.start = start
        self.pos = list(start)
        self.dir = (1,0,0)   # start facing +X
        self.stack = []
        self.nodes = set()
        self.rooms = {}  # (x,y,z) -> kind
        self.graph = defaultdict(set)
        self.stairs = set()

    def inb(self, x,y,z): return 0 <= x < self.w and 0 <= y < self.h and 0 <= z <= self.zmax
    def carve(self, x,y,z):
        if self.inb(x,y,z):
            self.grid[z][y][x] = FLOOR
            self.nodes.add((x,y,z))

    def connect(self, a,b):
        self.graph[a].add(b); self.graph[b].add(a)

    def yaw(self, right=True):
        dx,dy,dz = self.dir
        if dz != 0:  # keep vertical heading unchanged
            return
        order = [(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)]
        i = order.index((dx,dy,0))
        ni = (i + (1 if right else -1)) % 4
        self.dir = (order[ni][0], order[ni][1], 0)

    def pitch(self, up=True):
        dx,dy,dz = self.dir
        if dz == 0:
            self.dir = (0,0, 1 if up else -1)
        else:
            # If you prefer "restore last yaw" store it on push/pop; here we reset to +X
            self.dir = (1,0,0)

    def forward(self):
        x,y,z = self.pos
        nx,ny,nz = x + self.dir[0], y + self.dir[1], z + self.dir[2]
        if self.inb(nx,ny,nz):
            self.carve(x,y,z); self.carve(nx,ny,nz)
            self.connect((x,y,z),(nx,ny,nz))
            self.pos = [nx,ny,nz]

    def place_room(self, kind, radius=1):
        x,y,z = self.pos
        for yy in range(y-radius, y+radius+1):
            for xx in range(x-radius, x+radius+1):
                if self.inb(xx,yy,z):
                    self.carve(xx,yy,z)
        self.rooms[tuple(self.pos)] = kind

    def stair_up(self):
        x,y,z = self.pos
        if z >= self.zmax: return
        self.carve(x,y,z); self.carve(x,y,z+1)
        self.connect((x,y,z),(x,y,z+1))
        self.pos = [x,y,z+1]
        self.stairs.update({(x,y,z),(x,y,z+1)})

    def stair_down(self):
        x,y,z = self.pos
        if z <= 0: return
        self.carve(x,y,z); self.carve(x,y,z-1)
        self.connect((x,y,z),(x,y,z-1))
        self.pos = [x,y,z-1]
        self.stairs.update({(x,y,z),(x,y,z-1)})

    def push(self): self.stack.append((tuple(self.pos), self.dir))
    def pop(self):
        if self.stack:
            (x,y,z), d = self.stack.pop()
            self.pos = [x,y,z]; self.dir = d

    def render(self, s, stair_limit_per_level=8):
        # make start explicit
        self.carve(*self.start)
        self.graph[self.start]  # touch key

        stair_budget = [0]*(self.zmax+1)

        i = 0
        while i < len(s):
            ch = s[i]
            if ch == 'C': self.forward()
            elif ch == 'R': self.place_room('R', radius=1)
            elif ch == 'T': self.place_room('T', radius=1)
            elif ch == 'K': self.place_room('K', radius=1)
            elif ch == 'B': self.place_room('B', radius=1)
            elif ch == '+': self.yaw(True)
            elif ch == '-': self.yaw(False)
            elif ch == '{': self.pitch(True)
            elif ch == '}': self.pitch(False)
            elif ch == '[': self.push()
            elif ch == ']': self.pop()
            elif ch == 'U':
                x,y,z = self.pos
                if z < self.zmax and stair_budget[z] < stair_limit_per_level and stair_budget[z+1] < stair_limit_per_level:
                    self.stair_up()
                    stair_budget[z]   += 1
                    stair_budget[z+1] += 1
            elif ch == 'D':
                x,y,z = self.pos
                if z > 0 and stair_budget[z] < stair_limit_per_level and stair_budget[z-1] < stair_limit_per_level:
                    self.stair_down()
                    stair_budget[z]   += 1
                    stair_budget[z-1] += 1
            # ';' '.' ignored
            i += 1

    def farthest_node(self, source=None):
        if not self.graph: return (tuple(self.pos),0)
        s = source or tuple(self.start)
        q = deque([s]); dist = {s:0}
        while q:
            u = q.popleft()
            for v in self.graph[u]:
                if v not in dist:
                    dist[v] = dist[u]+1
                    q.append(v)
        v = max(dist, key=dist.get)
        return v, dist[v]

# ---------------- Pipeline ----------------
def generate_3d(iterations=7, size=(60,40), zmax=4, seed=42,
                max_stair_run=2, grammar_path=None):
    random.seed(seed)

    if grammar_path:
        axiom, rules = load_grammar(grammar_path)
    else:
        axiom, rules = 'S', ruleset3d()

    ls = LSystem(axiom, rules)
    plan = ls.iterate(iterations)
    plan = sanitize_plan(plan, zmin=0, zmax=zmax, max_run_stairs=max_stair_run)

    W,H = size
    d = Dungeon3D(W,H,zmax, start=(W//2, H//2, 0))
    d.render(plan)

    # Post: farthest 3D for boss/key
    bxy, _ = d.farthest_node(d.start)
    d.rooms[bxy] = 'B'
    kxy, _ = d.farthest_node(bxy)
    if kxy != bxy:
        d.rooms[kxy] = 'K'
    return d, plan


if __name__ == "__main__":
    d, plan = generate_3d(iterations=20, size=(54,36), zmax=4, seed=42)
    for z in range(d.zmax+1):
        print(f"\n=== LEVEL z={z} ===")
        for y in range(d.h):
            row = []
            for x in range(d.w):
                ch = '#'
                if d.grid[z][y][x] == FLOOR: ch = '.'
                if (x,y,z) in d.rooms:
                    ch = {'R':'r','T':'t','B':'B','K':'k'}[d.rooms[(x,y,z)]]
                if (x,y,z) in d.stairs: ch = 'S'
                row.append(ch)
            print(''.join(row))
    print("\nPlan length:", len(plan), "Nodes:", len(d.nodes), "Rooms:", len(d.rooms), "Graph nodes:", len(d.graph))
