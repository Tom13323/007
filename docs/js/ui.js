/*
 * 007 v10.2 website - the screens: the pygame interface of 007-premode-pygame-v10.2.py redrawn on a
 * 1280 x 720 <canvas>, with the same layout, colours, timings and controls.  Each screen is an async
 * function that waits for the next frame with `await frame()` (pygame's clock.tick + display.flip).
 * Test options in the address: autoplay=1, fast=1, skiptutorial=1, games=N, mode=1|2|4,
 * difficulty=easy|normal|hard, timer=3|5|10, seed=N, storage=memory, mouse=x,y, page=2, timerframes=1,
 * hold=gameover (see README.md).
 */
(function () {
  "use strict";
  const C = window.Core;
  // game coordinates (the canvas is scaled to the window)
  const W = 1280, H = 720;

  // ============================================================== options
  const Q = new URLSearchParams(location.search);
  // options from the address, for testing (see the top of the file)
  const OPTS = {
    autoplay: Q.get("autoplay") === "1", fast: Q.get("fast") === "1", skipTutorial: Q.get("skiptutorial") === "1",
    games: Number(Q.get("games") || 0), mode: Number(Q.get("mode") || 4), difficulty: Q.get("difficulty") || "normal",
    timer: Number(Q.get("timer") || 10), seed: Q.get("seed") === null ? null : Number(Q.get("seed")),
    storage: Q.get("storage") || "browser",
  };
  // animation speed factor (4x with fast=1)
  const speed = () => (OPTS.fast ? 4.0 : 1.0);
  // time in seconds
  const now = () => performance.now() / 1000;

  // ============================================================== colours, names
  const C_BG = [11, 13, 30], C_PANEL = [22, 26, 54], C_PANEL2 = [32, 38, 78], C_TEXT = [240, 240, 250];
  const C_MUTED = [150, 158, 190], C_GOLD = [255, 205, 70], C_GREEN = [70, 225, 140], C_RED = [245, 75, 90];
  const C_BLUE = [95, 175, 255], C_PURPLE = [190, 125, 255], C_ORANGE = [255, 150, 60];
  // colour, name, key and button order of each action
  const ACT_COLOR = { reload: [120, 205, 255], shoot: [255, 205, 70], defense: [90, 225, 150], bomb: [255, 150, 60],
    reflect: [190, 125, 255], mg: [255, 85, 95] };
  const ACT_NAME = { reload: "RELOAD", shoot: "SHOOT", defense: "DEFENSE", bomb: "BOMB", reflect: "REFLECT", mg: "MACHINE GUN" };
  const ACT_KEY = { reload: "R", shoot: "S", defense: "D", bomb: "B", reflect: "E", mg: "M" };
  const ACT_ORDER = ["reload", "shoot", "defense", "bomb", "reflect", "mg"];
  // colour of each seat
  const AVATAR_COLORS = [[70, 175, 255], [255, 145, 70], [185, 115, 255], [70, 205, 195], [255, 105, 165]];
  // 'MACHINE GUN' -> 'Machine Gun'
  const title = (s) => s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  // [r, g, b] -> CSS colour
  const rgb = (c, a = 1) => (a >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`);
  // the point a fraction t of the way from a to b
  const lerp = (a, b, t) => a + (b - a) * t;
  // mix two colours
  const lerpCol = (c1, c2, t) => [0, 1, 2].map((i) => Math.trunc(lerp(c1[i], c2[i], t)));
  // animation curve: fast start, soft stop
  const easeOut = (t) => { t = Math.max(0, Math.min(1, t)); return 1 - Math.pow(1 - t, 3); };
  // random integer from a to b
  const randint = (a, b) => a + Math.floor(Math.random() * (b - a + 1));

  // ============================================================== canvas
  const canvas = document.getElementById("game");
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  canvas.width = W * DPR; canvas.height = H * DPR;
  const g = canvas.getContext("2d");
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  // scale the canvas to the window, keeping 16:9
  function fit() {
    const s = Math.min(window.innerWidth / W, window.innerHeight / H);
    canvas.style.width = Math.floor(W * s) + "px";
    canvas.style.height = Math.floor(H * s) + "px";
  }
  window.addEventListener("resize", fit);
  fit();
  // one frame (like pygame's clock.tick(60)); &timerframes=1 uses a 16 ms timer instead, for headless tests
  const frame = Q.get("timerframes") === "1" ? () => new Promise((resolve) => setTimeout(resolve, 16))
    : () => new Promise((resolve) => requestAnimationFrame(resolve));

  // ============================================================== rectangles (like pygame.Rect)
  class Rect {
    // a rectangle like pygame.Rect
    constructor(x, y, w, h) { this.x = x; this.y = y; this.w = w; this.h = h; }
    // edges and centre, like pygame.Rect
    get right() { return this.x + this.w; } get bottom() { return this.y + this.h; }
    get centerx() { return this.x + Math.floor(this.w / 2); } get centery() { return this.y + Math.floor(this.h / 2); }
    get center() { return [this.centerx, this.centery]; }
    set center(p) { this.x = p[0] - Math.floor(this.w / 2); this.y = p[1] - Math.floor(this.h / 2); }
    // is the point inside (right and bottom edges excluded, like pygame)
    collidepoint(x, y) { if (Array.isArray(x)) [x, y] = x; return x >= this.x && x < this.right && y >= this.y && y < this.bottom; }
    // a copy of this rectangle
    copy() { return new Rect(this.x, this.y, this.w, this.h); }
    // grow around the centre (pygame's inflate_ip)
    inflate(dx, dy) { this.x -= Math.floor(dx / 2); this.y -= Math.floor(dy / 2); this.w += dx; this.h += dy; return this; }
  }

  // ============================================================== drawing helpers
  const FONTS = { title: [62, true], big: [40, true], med: [27, true], body: [22, false], bodyb: [22, true],
    small: [18, false], tiny: [15, false], num: [30, true] };
  const FAMILY = '"Segoe UI", "Helvetica Neue", Arial, sans-serif';
  // CSS font for one of the game's font names
  const fontCss = (f) => `${FONTS[f][1] ? "bold " : ""}${FONTS[f][0]}px ${FAMILY}`;
  // width, ascent and height of a text
  function measure(f, s) {
    g.font = fontCss(f);
    const m = g.measureText(s);
    const sz = FONTS[f][0];
    const asc = m.fontBoundingBoxAscent !== undefined ? m.fontBoundingBoxAscent : sz * 1.06;
    const desc = m.fontBoundingBoxDescent !== undefined ? m.fontBoundingBoxDescent : sz * 0.28;
    return { w: m.width, asc: asc, h: asc + desc };
  }
  // place a rectangle so that its anchor point is at pos
  function anchorRect(r, anchor, pos) {
    const [x, y] = pos;
    const hw = Math.floor(r.w / 2), hh = Math.floor(r.h / 2);
    const A = { topleft: [x, y], midtop: [x - hw, y], center: [x - hw, y - hh], midleft: [x, y - hh],
      midright: [x - r.w, y - hh], topright: [x - r.w, y], midbottom: [x - hw, y - r.h] }[anchor];
    r.x = A[0]; r.y = A[1];
    return r;
  }
  /** Draw a text at a position given by one point of its box (like the Python text()). */
  function text(f, s, pos, color = C_TEXT, anchor = "topleft", shadow = false, alpha = 255) {
    s = String(s);
    const m = measure(f, s);
    const r = anchorRect(new Rect(0, 0, Math.ceil(m.w), Math.ceil(m.h)), anchor, pos);
    g.font = fontCss(f);
    g.textBaseline = "alphabetic";
    if (shadow) { g.fillStyle = rgb([0, 0, 0], Math.min(alpha, 140) / 255); g.fillText(s, r.x + 2, r.y + 2 + m.asc); }
    g.fillStyle = rgb(color, alpha / 255);
    g.fillText(s, r.x, r.y + m.asc);
    return r;
  }
  // split a text into lines that fit `width` pixels
  function wrap(f, s, width) {
    const words = s.split(/\s+/).filter(Boolean), lines = [];
    let cur = "";
    for (const w of words) {
      const t = (cur + " " + w).trim();
      if (measure(f, t).w <= width) cur = t; else { lines.push(cur); cur = w; }
    }
    if (cur) lines.push(cur);
    return lines;
  }
  // path of a rounded rectangle
  function rrPath(s, x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    s.beginPath();
    s.moveTo(x + r, y);
    s.arcTo(x + w, y, x + w, y + h, r); s.arcTo(x + w, y + h, x, y + h, r);
    s.arcTo(x, y + h, x, y, r); s.arcTo(x, y, x + w, y, r);
    s.closePath();
  }
  /** Rounded rectangle, filled or (width > 0) as an outline inside the rectangle. */
  function rrect(s, r, color, radius = 14, width = 0) {
    if (width > 0) {
      rrPath(s, r.x + width / 2, r.y + width / 2, r.w - width, r.h - width, radius - width / 2);
      s.lineWidth = width; s.strokeStyle = rgb(color); s.stroke();
    } else { rrPath(s, r.x, r.y, r.w, r.h, radius); s.fillStyle = rgb(color); s.fill(); }
  }
  // filled circle, or a ring of the given width inside the radius (like pygame)
  function circle(s, color, cx, cy, r, width = 0) {
    if (r <= 0) return;
    s.beginPath();
    if (width > 0 && width < r) { s.arc(cx, cy, r - width / 2, 0, Math.PI * 2); s.lineWidth = width; s.strokeStyle = rgb(color); s.stroke(); }
    else { s.arc(cx, cy, r, 0, Math.PI * 2); s.fillStyle = rgb(color); s.fill(); }
  }
  // straight line
  function line(s, color, p0, p1, width = 1) {
    s.beginPath(); s.moveTo(p0[0], p0[1]); s.lineTo(p1[0], p1[1]);
    s.lineWidth = width; s.lineCap = "butt"; s.strokeStyle = rgb(color); s.stroke();
  }
  // filled polygon, or its outline
  function polygon(s, color, pts, width = 0) {
    s.beginPath(); s.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) s.lineTo(pts[i][0], pts[i][1]);
    s.closePath();
    if (width > 0) { s.lineWidth = width; s.lineJoin = "miter"; s.strokeStyle = rgb(color); s.stroke(); }
    else { s.fillStyle = rgb(color); s.fill(); }
  }
  /** pygame.draw.arc: angles counter-clockwise on screen, the line drawn inside the box. */
  function arc(s, color, r, start, stop, width) {
    s.beginPath();
    s.ellipse(r.x + r.w / 2, r.y + r.h / 2, Math.max(0.5, r.w / 2 - width / 2), Math.max(0.5, r.h / 2 - width / 2), 0, -start, -stop, true);
    s.lineWidth = width; s.lineCap = "butt"; s.strokeStyle = rgb(color); s.stroke();
  }
  // filled ellipse in a box
  function ellipse(s, color, x, y, w, h) {
    s.beginPath(); s.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2); s.fillStyle = rgb(color); s.fill();
  }
  // glow: like pygame, six circles on a transparent surface, each one REPLACING the pixels of the larger one
  const GLOW_CACHE = new Map();
  // soft round glow (each size and colour drawn once, then reused)
  function glow(s, pos, radius, color, alpha = 90) {
    radius = Math.floor(radius);
    if (radius <= 0 || alpha <= 0) return;
    const key = radius + "|" + color.join(",") + "|" + Math.floor(alpha);
    let oc = GLOW_CACHE.get(key);
    if (!oc) {
      if (GLOW_CACHE.size > 600) GLOW_CACHE.clear();
      oc = document.createElement("canvas");
      oc.width = oc.height = radius * 2;
      const o = oc.getContext("2d");
      for (let i = 6; i > 0; i--) {
        o.save(); o.beginPath(); o.arc(radius, radius, Math.floor(radius * i / 6), 0, Math.PI * 2); o.clip();
        o.clearRect(0, 0, radius * 2, radius * 2);
        o.fillStyle = rgb(color, Math.floor(alpha * i / 6 / 2) / 255); o.fillRect(0, 0, radius * 2, radius * 2);
        o.restore();
      }
      GLOW_CACHE.set(key, oc);
    }
    s.drawImage(oc, pos[0] - radius, pos[1] - radius);
  }
  // fill the whole screen with one colour
  function fillScreen(color) { g.fillStyle = rgb(color); g.fillRect(0, 0, W, H); }

  /** Draw the icon of an action (same shapes as the Python draw_icon). */
  function drawIcon(s, kind, cx, cy, r, color) {
    const lw = Math.max(2, Math.trunc(r * 0.16));
    const dark = [20, 20, 40];
    if (kind === "reload") {
      arc(s, color, new Rect(cx - r * 0.75, cy - r * 0.75, r * 1.5, r * 1.5), 0.7, 5.5, lw + 1);
      const a = 0.7, px = cx + 0.75 * r * Math.cos(a), py = cy - 0.75 * r * Math.sin(a);
      polygon(s, color, [[px + r * 0.45, py - r * 0.05], [px - r * 0.1, py - r * 0.4], [px + r * 0.05, py + r * 0.35]]);
      rrect(s, new Rect(cx - r * 0.12, cy - r * 0.3, r * 0.24, r * 0.6), color, 2);
    } else if (kind === "shoot" || kind === "mg") {
      const n = kind === "mg" ? 3 : 1;
      for (let i = 0; i < n; i++) {
        const oy = cy + (i - (n - 1) / 2) * r * 0.62;
        rrect(s, new Rect(cx - r * 0.8, oy - r * 0.2, r * 1.0, r * 0.4), color, 3);
        polygon(s, color, [[cx + r * 0.2, oy - r * 0.2], [cx + r * 0.85, oy], [cx + r * 0.2, oy + r * 0.2]]);
        line(s, dark, [cx - r * 0.45, oy - r * 0.2], [cx - r * 0.45, oy + r * 0.2], 2);
      }
    } else if (kind === "defense") {
      const pts = [[cx - r * 0.75, cy - r * 0.75], [cx + r * 0.75, cy - r * 0.75], [cx + r * 0.75, cy + r * 0.1],
        [cx, cy + r * 0.9], [cx - r * 0.75, cy + r * 0.1]];
      polygon(s, color, pts); polygon(s, dark, pts, 2);
      line(s, dark, [cx, cy - r * 0.6], [cx, cy + r * 0.6], 2);
    } else if (kind === "bomb") {
      circle(s, color, Math.trunc(cx - r * 0.05), Math.trunc(cy + r * 0.2), Math.trunc(r * 0.62));
      line(s, color, [cx + r * 0.3, cy - r * 0.3], [cx + r * 0.6, cy - r * 0.7], lw);
      for (const ang of [0, 1.2, 2.4, 3.6, 4.8]) {
        line(s, C_GOLD, [cx + r * 0.62, cy - r * 0.74], [cx + r * 0.62 + r * 0.28 * Math.cos(ang), cy - r * 0.74 + r * 0.28 * Math.sin(ang)], 2);
      }
    } else if (kind === "reflect") {
      line(s, color, [cx + r * 0.35, cy - r * 0.85], [cx + r * 0.35, cy + r * 0.85], lw + 2);
      line(s, color, [cx - r * 0.85, cy - r * 0.7], [cx + r * 0.3, cy], lw);
      line(s, color, [cx + r * 0.3, cy], [cx - r * 0.7, cy + r * 0.65], lw);
      polygon(s, color, [[cx - r * 0.9, cy + r * 0.75], [cx - r * 0.35, cy + r * 0.7], [cx - r * 0.7, cy + r * 0.25]]);
    }
  }

  // ============================================================== input
  const queue = [];
  // last mouse position
  let mouse = [0, 0];
  if (Q.get("mouse")) mouse = Q.get("mouse").split(",").map(Number);
  // mouse position in game coordinates
  function toLogical(e) {
    const r = canvas.getBoundingClientRect();
    return [Math.floor((e.clientX - r.left) * W / r.width), Math.floor((e.clientY - r.top) * H / r.height)];
  }
  canvas.addEventListener("pointermove", (e) => { mouse = toLogical(e); queue.push({ type: "MOUSEMOTION", pos: mouse }); });
  canvas.addEventListener("pointerdown", (e) => {
    mouse = toLogical(e);
    queue.push({ type: "MOUSEBUTTONDOWN", button: e.button === 2 ? 3 : (e.button === 0 ? 1 : 2), pos: mouse });
    e.preventDefault();
  });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  /** The physical key, so an input method (for example Chinese) cannot swallow it. */
  function keyOf(e) {
    const c = e.code || "";
    if (c.startsWith("Key")) return c.slice(3).toLowerCase();
    if (c.startsWith("Digit")) return c.slice(5);
    if (/^Numpad\d$/.test(c)) return c.slice(6);
    return { Space: "space", Enter: "return", NumpadEnter: "return", Escape: "escape", ArrowRight: "right" }[c] || null;
  }
  window.addEventListener("keydown", (e) => {
    const k = keyOf(e);
    if (!k || e.ctrlKey || e.metaKey || e.altKey) return;
    queue.push({ type: "KEYDOWN", key: k });
    e.preventDefault();
  });
  // take all queued input events
  const drain = () => queue.splice(0, queue.length);
  // is e a press of one of these keys
  const isKey = (e, ...keys) => e.type === "KEYDOWN" && keys.includes(e.key);

  // thrown when the player leaves the match with MENU
  class ReturnToMenu extends Error {}
  // thrown when Esc stops a replay
  class ReplayStop extends Error {}

  // ============================================================== shared state
  const ctx = {
    cfg: {}, log: [], insight: {}, nextInsight: {}, botNames: new Map(), showP: false, pbtnShown: false,
    inMatch: false, btnShown: false, paused: 0.0, store: null, stats: C.BotStore.summary([]), replaying: false,
    secretRect: null, secretFound: false, roundNo: 0,
  };
  // the MENU button and the SHOW P strip
  const MENU_BTN = new Rect(W - 134, 56, 110, 32);
  const PROB_STRIP = new Rect(16, 432, 380, 38);
  const PROB_BTN = new Rect(16 + 380 - 112, 436, 100, 30);
  const PROB_ROW_H = 76;
  // where the players stand; AV_R = avatar radius
  const ARENA_X0 = 430, ARENA_X1 = 1230, AV_R = 44;

  // ============================================================== table, header, log, buttons
  function layout(players, viewerId) {
    const pos = {};
    const others = players.filter((p) => p.id !== viewerId);
    const k = others.length;
    others.forEach((p, i) => {
      pos[p.id] = k === 1 ? [(ARENA_X0 + ARENA_X1) / 2, 190] : [ARENA_X0 + (ARENA_X1 - ARENA_X0) * (i + 0.5) / k, 190];
    });
    pos[viewerId] = [(ARENA_X0 + ARENA_X1) / 2, 430];
    return pos;
  }
  // fixed colour of a seat
  const avatarColor = (p) => AVATAR_COLORS[p.id % AVATAR_COLORS.length];

  // all players with bullets, Reflect status, target rings, number badges and action bubbles
  function drawTable(players, pos, o = {}) {
    const s = g, tnow = o.tnow || 0;
    for (const p of players) {
      const pid = p.id;
      let [x, y] = pos[pid];
      if (o.shake && o.shake[pid]) { const amp = o.shake[pid]; x += Math.sin(tnow * 70) * amp; y += Math.cos(tnow * 55) * amp * 0.5; }
      const alive = !o.aliveVis ? p.alive : (pid in o.aliveVis ? o.aliveVis[pid] : p.alive);
      const col = alive ? avatarColor(p) : [70, 72, 90];
      if (alive) glow(s, [x, y], AV_R + 26, col, 70);
      circle(s, col, Math.trunc(x), Math.trunc(y), AV_R);
      circle(s, alive ? [255, 255, 255] : [110, 112, 130], Math.trunc(x), Math.trunc(y), AV_R, 3);
      const dark = alive ? [18, 20, 42] : [40, 42, 58];
      circle(s, dark, Math.trunc(x), Math.trunc(y - 9), 13);
      ellipse(s, dark, x - 23, y + 5, 46, 34);
      if (!alive) { line(s, C_RED, [x - 26, y - 26], [x + 26, y + 26], 7); line(s, C_RED, [x + 26, y - 26], [x - 26, y + 26], 7); }
      text("small", p.name, [x, y + AV_R + 6], alive ? C_TEXT : C_MUTED, "midtop", true);
      const b = o.bulletsVis ? (pid in o.bulletsVis ? o.bulletsVis[pid] : p.bullets) : p.bullets;
      const by = y + AV_R + 34;
      const bc = b >= C.MG_COST ? C_RED : (b >= 7 ? C_GOLD : C_TEXT);
      if (alive) {
        drawIcon(s, "shoot", Math.trunc(x - 28), Math.trunc(by + 9), 13, bc);
        text("med", `${b}`, [x - 10, by - 6], bc);
        drawIcon(s, "defense", Math.trunc(x + 32), Math.trunc(by + 9), 10, p.has_reflect ? C_PURPLE : [75, 78, 100]);   // Reflect still available?
        if (b >= C.MG_COST) text("tiny", "MG READY", [x + 50, by + 9], C_RED, "midleft");
      }
      if (o.rings && o.rings[pid] && alive) {
        const rc = o.rings[pid], rad = AV_R + 10 + 4 * Math.sin(tnow * 9);
        circle(s, rc, Math.trunc(x), Math.trunc(y), Math.trunc(rad), 4);
        glow(s, [x, y], Math.trunc(rad) + 14, rc, 110);
      }
      if (o.badges && o.badges[pid]) {
        circle(s, [0, 0, 0], Math.trunc(x + 34), Math.trunc(y - 34), 14);
        text("small", o.badges[pid], [x + 34, y - 34], C_TEXT, "center");
      }
      if (o.bubbles && o.bubbles[pid] && alive) {
        const [act, sub, sc] = o.bubbles[pid];
        if (sc > 0) {
          const size = Math.trunc(34 * sc), cx = Math.trunc(x), cy = Math.trunc(y - AV_R - 36);
          const rect = new Rect(0, 0, size * 2, size * 2);
          rect.center = [cx, cy];
          const c = ACT_COLOR[act];
          rrect(s, rect, [24, 28, 60], 14); rrect(s, rect, c, 14, 3);
          drawIcon(s, act, cx, cy - 3, Math.trunc(size * 0.62), c);
          if (sc >= 0.95 && sub) text("tiny", `> ${sub}`, [cx, cy - size - 2], c, "midbottom", true);
        }
      }
    }
  }
  // background, round number, mode label and MENU button
  function drawHeader(roundNo, rightText = "") {
    fillScreen(C_BG);
    g.fillStyle = rgb([13, 15, 34]);
    for (let i = 0; i < H; i += 4) g.fillRect(0, i, W, 1);                  // subtle scanlines
    if (roundNo) text("med", `ROUND ${roundNo}`, [24, 22], C_GOLD);
    if (rightText) text("small", rightText, [W - 24, 28], C_MUTED, "topright");
    if (ctx.inMatch) {
      const hot = MENU_BTN.collidepoint(mouse);
      rrect(g, MENU_BTN, hot ? [70, 40, 60] : C_PANEL, 10);
      text("small", "< MENU", MENU_BTN.center, hot ? C_TEXT : C_MUTED, "center");
      ctx.btnShown = true;
    }
  }
  // side log and, with SHOW P, the bots' probabilities
  function drawLog() {
    let bottom = 470, pp = null;
    if (ctx.botNames.size) {
      bottom = PROB_STRIP.y - 6;
      if (ctx.showP) {
        const nb = ctx.botNames.size;
        const ph = 38 + (nb > 1 ? Math.floor((nb + 1) / 2) : 1) * PROB_ROW_H;
        pp = new Rect(16, PROB_STRIP.y - 6 - ph, 380, ph);
        bottom = pp.y - 6;
      }
    }
    const panel = new Rect(16, 70, 380, bottom - 70);
    rrect(g, panel, C_PANEL, 16);
    text("small", "WHAT JUST HAPPENED", [panel.x + 16, panel.y + 12], C_MUTED);
    const lines = [];
    for (const [ln0, col] of ctx.log.slice(-40)) for (const ln of wrap("tiny", ln0, panel.w - 32).slice(0, 2)) lines.push([ln, col]);
    const maxLines = Math.max(1, Math.floor((panel.bottom - 12 - (panel.y + 42)) / 19));
    let y = panel.y + 42;
    for (const [ln, col] of lines.slice(-maxLines)) { text("tiny", ln, [panel.x + 16, y], col); y += 19; }
    if (!ctx.botNames.size) return;
    if (pp) {
      rrect(g, pp, C_PANEL, 16);
      text("small", "BOT PROBABILITIES (last round)", [pp.x + 16, pp.y + 8], C_MUTED);
      const cols = ctx.botNames.size === 1 ? 1 : 2, cw = Math.floor((pp.w - 24) / cols);
      let k = 0;
      for (const [pid, name] of ctx.botNames) {
        const cx = pp.x + 16 + (k % cols) * cw, cy = pp.y + 32 + Math.floor(k / cols) * PROB_ROW_H;
        k++;
        text("tiny", name, [cx, cy], avatarColor({ id: pid }));
        const mix = ctx.insight[pid];
        if (mix === undefined) { text("tiny", Object.keys(ctx.insight).length ? "eliminated" : "(after round 1)", [cx + 8, cy + 18], C_MUTED); continue; }
        mix.slice(0, 3).forEach(([a, p], j) => {
          text("tiny", `${ACT_NAME[a] || a}  ${String(Math.round(p * 100)).padStart(3, " ")}%`, [cx + 8, cy + 18 + j * 17], ACT_COLOR[a] || C_TEXT);
        });
      }
    }
    rrect(g, PROB_STRIP, C_PANEL, 16);
    text("tiny", "Bot probabilities", [PROB_STRIP.x + 16, PROB_STRIP.centery], C_MUTED, "midleft");
    const hot = PROB_BTN.collidepoint(mouse);
    rrect(g, PROB_BTN, hot ? C_PANEL2 : [28, 33, 68], 10);
    rrect(g, PROB_BTN, C_PURPLE, 10, 2);
    text("small", ctx.showP ? "HIDE P" : "SHOW P", PROB_BTN.center, C_PURPLE, "center");
    ctx.pbtnShown = true;
  }
  // banner text: who was eliminated
  function elimText(names) {
    if (names.length === 1) return names[0] === "You" ? `${names[0]} are eliminated` : `${names[0]} eliminated`;
    return names.slice(0, -1).join(", ") + ` and ${names[names.length - 1]} ` + (names.includes("You") ? "are eliminated" : "eliminated");
  }
  // a round in plain words for the side log
  function describeEvents(players, outcome) {
    const names = {};
    for (const p of players) names[p.id] = p.name;
    const lines = [], acts = outcome.actions;
    for (const pid of Object.keys(acts)) {
      const a = acts[pid];
      const extra = a.target !== null && a.target !== undefined ? ` -> ${names[a.target]}` : "";
      lines.push([`${names[pid]}: ${ACT_NAME[a.action]}${extra}`, C_TEXT]);
    }
    for (const [k, a, t] of outcome.events) {
      if (k === "hit") lines.push([`${names[a]} hit ${names[t]}`, C_RED]);
      else if (k === "blocked") lines.push([`${names[t]} blocked ${names[a]}`, C_GREEN]);
      else if (k === "reflected") lines.push([`${names[t]} reflected ${names[a]}'s bullet - ${names[a]} dies`, C_PURPLE]);
      else if (k === "cancel") lines.push([`${names[a]} and ${names[t]}: bullets cancelled`, C_BLUE]);
      else if (k === "mg") lines.push([`${names[a]}'s Machine Gun killed ${names[t]}`, C_RED]);
      else if (k === "mg_clash") lines.push([`${names[a]} and ${names[t]}: MG clash, both die`, C_RED]);
    }
    for (const pid of outcome.deaths) lines.push([elimText([names[pid]]), C_RED]);
    if (!outcome.deaths.length) lines.push(["Nobody died.", C_GREEN]);
    return lines;
  }
  // positions of the six action buttons
  function buttonRects() {
    const w = 178, h = 150, gap = 14, x0 = Math.floor((W - (6 * w + 5 * gap)) / 2);
    const out = {};
    ACT_ORDER.forEach((a, i) => { out[a] = new Rect(x0 + i * (w + gap), 538, w, h); });
    return out;
  }
  // [can be used, cost label, reason it cannot]
  function buttonState(a, me, players) {
    const b = me.bullets, bc = C.bombCost(players);
    if (a === "reload") return [true, "+1 bullet", ""];
    if (a === "defense") return [true, "free", ""];
    if (a === "shoot") return [b >= 1, "1 bullet", b < 1 ? "need 1" : ""];
    if (a === "bomb") return [b >= bc, `${bc} bullet${bc > 1 ? "s" : ""}`, b < bc ? `need ${bc}` : ""];
    if (a === "reflect") return [me.has_reflect, "once per game", !me.has_reflect ? "used" : ""];
    return [b >= C.MG_COST, "10 bullets", b < C.MG_COST ? "need 10" : ""];
  }
  // the six action buttons
  function drawButtons(me, players, hover = null, selected = null) {
    const rects = buttonRects();
    for (const a of ACT_ORDER) {
      const [ok, cost, why] = buttonState(a, me, players);
      const c = ACT_COLOR[a];
      let face = ok ? [28, 33, 68] : [24, 26, 40];
      if (hover === a && ok) face = [44, 52, 100];
      const rr = rects[a].copy();
      if (selected === a) { rr.inflate(8, 8); glow(g, rr.center, 100, c, 110); }
      rrect(g, rr, face, 16);
      rrect(g, rr, ok ? c : [62, 64, 84], 16, selected === a ? 4 : 2);
      drawIcon(g, a, rr.centerx, rr.y + 58, 32, ok ? c : [70, 72, 92]);
      text("body", ACT_NAME[a], [rr.centerx, rr.y + 102], ok ? C_TEXT : [85, 88, 110], "midtop");
      text("tiny", (why && !ok) ? why : cost, [rr.centerx, rr.y + 128], ok ? C_MUTED : C_RED, "midtop");
      const kb = new Rect(rr.x + 8, rr.y + 8, 24, 24);
      rrect(g, kb, [10, 12, 26], 6);
      text("tiny", ACT_KEY[a], kb.center, ok ? c : [70, 72, 92], "center");
    }
  }

  // ============================================================== dialogs and events
  async function confirmMenu() {
    const t0 = now();
    const snap = document.createElement("canvas");
    snap.width = canvas.width; snap.height = canvas.height;
    snap.getContext("2d").drawImage(canvas, 0, 0);
    const box = new Rect(0, 0, 520, 220);
    box.center = [Math.floor(W / 2), Math.floor(H / 2)];
    const yes = new Rect(box.x + 40, box.bottom - 80, 200, 52), no = new Rect(box.right - 240, box.bottom - 80, 200, 52);
    let answer = null;
    while (answer === null) {
      await frame();
      for (const e of drain()) {
        if (e.type === "MOUSEBUTTONDOWN" && e.button === 1) { if (yes.collidepoint(e.pos)) answer = true; else if (no.collidepoint(e.pos)) answer = false; }
        if (isKey(e, "return", "y")) answer = true; else if (isKey(e, "escape", "n")) answer = false;
      }
      g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(snap, 0, 0); g.restore();
      g.fillStyle = "rgba(0,0,0,0.667)"; g.fillRect(0, 0, W, H);
      rrect(g, box, C_PANEL, 18);
      text("med", "End this match?", [box.centerx, box.y + 28], C_GOLD, "midtop");
      text("body", "The game is terminated and you return to the game selection screen.", [box.centerx, box.y + 80], C_MUTED, "midtop");
      rrect(g, yes, yes.collidepoint(mouse) ? [150, 50, 60] : [110, 40, 50], 12);
      rrect(g, no, no.collidepoint(mouse) ? [60, 90, 70] : [40, 70, 55], 12);
      text("body", "Yes, leave (Y)", yes.center, C_TEXT, "center");
      text("body", "No, keep playing (N)", no.center, C_TEXT, "center");
    }
    ctx.paused += now() - t0;
    return answer;
  }
  /** This frame's events, after handling the ones that apply on every screen (MENU, SHOW P, Esc in a replay). */
  async function pumpEvents() {
    const evs = drain();
    const shown = ctx.btnShown, pshown = ctx.pbtnShown;
    ctx.btnShown = false; ctx.pbtnShown = false;
    const out = [];
    for (const e of evs) {
      if (e.type === "MOUSEBUTTONDOWN" && e.button === 1 && ctx.inMatch && pshown && PROB_BTN.collidepoint(e.pos)) { ctx.showP = !ctx.showP; continue; }
      if (e.type === "MOUSEBUTTONDOWN" && e.button === 1 && ctx.inMatch && shown && MENU_BTN.collidepoint(e.pos)) {
        if (await confirmMenu()) throw new ReturnToMenu();
        continue;
      }
      if (ctx.replaying && isKey(e, "escape")) throw new ReplayStop();
      out.push(e);
    }
    return out;
  }

  // ============================================================== the player's turn
  async function humanTurn(me, players, roundNo, hotseat = false) {
    const limit = ctx.cfg.timer, pos = layout(players, me.id);
    const opps = players.filter((p) => p.alive && p.id !== me.id);
    const btns = buttonRects();
    let pending = null, hover = null, autoStage = 0;
    const t0 = now();
    ctx.paused = 0.0;
    const rng = new C.Rng();
    const modeLabel = ctx.cfg.label;
    // package an action and its target
    const finish = (action, target = null) => ({ action: action, target: target });
    while (true) {
      await frame();
      const tn = now(), el = tn - t0 - ctx.paused, left = limit - el;
      for (const e of await pumpEvents()) {
        if (e.type === "MOUSEMOTION") {
          hover = null;
          for (const a of ACT_ORDER) if (btns[a].collidepoint(e.pos)) { hover = a; break; }
        } else if (e.type === "MOUSEBUTTONDOWN") {
          if (e.button === 3) pending = null;
          else if (e.button === 1) {
            if (pending) {
              for (const p of opps) {
                const [px, py] = pos[p.id];
                if ((e.pos[0] - px) ** 2 + (e.pos[1] - py) ** 2 <= (AV_R + 16) ** 2) return finish(pending, p.id);
              }
              pending = null;
            } else {
              for (const a of ACT_ORDER) {
                if (btns[a].collidepoint(e.pos) && buttonState(a, me, players)[0]) {
                  if (a === "shoot" || a === "mg") pending = a; else return finish(a);
                }
              }
            }
          }
        } else if (e.type === "KEYDOWN") {
          const k = e.key;
          if (k === "escape") pending = null;
          else if (pending && /^[1-9]$/.test(k)) { const i = Number(k) - 1; if (i < opps.length) return finish(pending, opps[i].id); }
          else {
            for (const a of ACT_ORDER) {
              if (k === ACT_KEY[a].toLowerCase() && buttonState(a, me, players)[0]) {
                if (a === "shoot" || a === "mg") pending = a; else return finish(a);
              }
            }
          }
        }
      }
      if (OPTS.autoplay) {
        if (autoStage === 0 && el > 0.35 / speed()) {
          const legal = C.legalActions(me, players);
          const wts = { reload: 3, defense: 1.5, shoot: 2.5, bomb: 1, reflect: 0.6, mg: 4 };
          const tot = legal.reduce((s, x) => s + wts[x], 0);
          let r = Math.random() * tot, a = legal[legal.length - 1];
          for (const x of legal) { r -= wts[x]; if (r < 0) { a = x; break; } }
          if (a === "shoot" || a === "mg") { pending = a; autoStage = 1; } else return finish(a);
        } else if (autoStage === 1 && el > 0.7 / speed()) return finish(pending, rng.choice(opps).id);
      }
      if (left <= 0) {
        if (me.bullets >= C.MG_COST) return finish("mg", rng.choice(opps).id);
        return finish("reload");
      }
      // ---- draw
      drawHeader(roundNo, modeLabel);
      const frac = Math.max(0, left / limit);
      const barcol = frac < 0.625 ? lerpCol(C_RED, C_GREEN, Math.min(1, frac * 1.6)) : C_GREEN;
      g.fillStyle = rgb([30, 34, 66]); g.fillRect(0, 0, W, 10);
      g.fillStyle = rgb(barcol); g.fillRect(0, 0, Math.trunc(W * frac), 10);
      const shake = left < 3 ? 3 : 0;
      text("title", `${Math.max(0, Math.ceil(left))}`, [W / 2 + randint(-shake, shake), 50], left < 3 ? C_RED : C_TEXT, "midtop", true);
      let rings = null, badges = null;
      if (pending) {
        rings = {}; badges = {};
        opps.forEach((p, i) => { rings[p.id] = pending === "mg" ? ACT_COLOR[pending] : C_BLUE; badges[p.id] = String(i + 1); });
      }
      drawTable(players, pos, { rings: rings, tnow: tn, badges: badges });
      drawLog();
      if (hotseat) text("med", `${me.name}'s turn`, [W / 2, 448], C_GOLD, "midtop");
      if (pending) {
        text("body", `Pick a target for ${ACT_NAME[pending]}: click a blinking player (or press 1-${opps.length}).  Esc cancels.`,
          [745, 515], pending === "mg" ? C_RED : C_BLUE, "midright");
      } else text("body", "Choose your action  -  your choice is final once you click.", [745, 515], C_MUTED, "midright");
      drawButtons(me, players, hover, pending);
    }
  }
  // Testing mode: 'pass the device' between two players
  async function passScreen(name, color) {
    const t0 = now();
    ctx.btnShown = false;
    while (true) {
      await frame();
      let go = false;
      for (const e of await pumpEvents()) if (e.type === "MOUSEBUTTONDOWN" || isKey(e, "return", "space")) go = true;
      if (OPTS.autoplay && now() - t0 > 0.2 / speed()) go = true;
      if (go) return;
      fillScreen(C_BG);
      glow(g, [W / 2, H / 2 - 40], 220, color, 120);
      text("med", "Pass the device to", [W / 2, 230], C_MUTED, "midtop");
      text("title", name, [W / 2, 270], color, "midtop", true);
      text("body", "Make sure the other players are not looking.  Click or press Enter when ready.", [W / 2, 400], C_TEXT, "midtop");
    }
  }

  // ============================================================== round animation
  function buildSchedule(pos, acts, outcome) {
    const T0 = 1.05, TR = 0.55, sched = [], death = {};
    let k = 0;
    // colour of a player's chosen action
    const colOf = (pid) => ACT_COLOR[acts[pid].action];
    // midpoint between two players
    const mid = (a, t) => [(pos[a][0] + pos[t][0]) / 2, (pos[a][1] + pos[t][1]) / 2];
    // remember the earliest moment a player dies in the animation
    const setDeath = (pid, t) => { death[pid] = Math.min(pid in death ? death[pid] : 99, t); };
    for (const pid of Object.keys(acts)) if (acts[pid].action === "bomb") sched.push({ k: "ring", t0: T0 - 0.15, dur: 0.8, p: pos[pid], col: ACT_COLOR.bomb });
    for (const [kind, a, t] of outcome.events) {
      const t0 = T0 + 0.07 * k;
      k++;
      if (kind === "hit") {
        sched.push({ k: "bullet", t0: t0, dur: TR, p0: pos[a], p1: pos[t], col: colOf(a) });
        sched.push({ k: "burst", t0: t0 + TR, dur: 0.6, p: pos[t], col: C_RED });
        setDeath(t, t0 + TR);
      } else if (kind === "blocked") {
        sched.push({ k: "bullet", t0: t0, dur: TR, p0: pos[a], p1: pos[t], col: colOf(a) });
        sched.push({ k: "shield", t0: t0 + TR, dur: 0.55, p: pos[t], col: ACT_COLOR.defense });
      } else if (kind === "reflected") {
        sched.push({ k: "bullet", t0: t0, dur: TR, p0: pos[a], p1: pos[t], col: colOf(a) });
        sched.push({ k: "shield", t0: t0 + TR, dur: 0.55, p: pos[t], col: ACT_COLOR.reflect });
        sched.push({ k: "bullet", t0: t0 + TR + 0.12, dur: TR, p0: pos[t], p1: pos[a], col: ACT_COLOR.reflect });
        sched.push({ k: "burst", t0: t0 + 2 * TR + 0.12, dur: 0.6, p: pos[a], col: C_PURPLE });
        setDeath(a, t0 + 2 * TR + 0.12);
      } else if (kind === "cancel") {
        const m = mid(a, t);
        sched.push({ k: "bullet", t0: t0, dur: TR, p0: pos[a], p1: m, col: colOf(a) });
        sched.push({ k: "bullet", t0: t0, dur: TR, p0: pos[t], p1: m, col: colOf(t) });
        sched.push({ k: "burst", t0: t0 + TR, dur: 0.7, p: m, col: C_BLUE });
      } else if (kind === "mg") {
        sched.push({ k: "beam", t0: t0, dur: 0.45, p0: pos[a], p1: pos[t], col: ACT_COLOR.mg });
        sched.push({ k: "burst", t0: t0 + 0.2, dur: 0.7, p: pos[t], col: C_RED });
        setDeath(t, t0 + 0.25);
      } else if (kind === "mg_clash") {
        sched.push({ k: "beam", t0: t0, dur: 0.45, p0: pos[a], p1: pos[t], col: ACT_COLOR.mg });
        sched.push({ k: "beam", t0: t0, dur: 0.45, p0: pos[t], p1: pos[a], col: ACT_COLOR.mg });
        sched.push({ k: "burst", t0: t0 + 0.2, dur: 0.8, p: mid(a, t), col: C_ORANGE });
        death[a] = death[t] = t0 + 0.25;
      }
    }
    const end = Math.max.apply(null, sched.map((s) => s.t0 + s.dur).concat([T0 + 0.4]));
    return [sched, death, end];
  }
  // draw the animation pieces visible at time t
  function drawSched(sched, t) {
    for (const it of sched) {
      const lt = t - it.t0;
      if (lt < 0 || lt > it.dur) continue;
      const f = lt / it.dur;
      if (it.k === "bullet") {
        const x = lerp(it.p0[0], it.p1[0], f), y = lerp(it.p0[1], it.p1[1], f);
        for (let i = 1; i < 6; i++) {
          const tf = Math.max(0, f - i * 0.04);
          circle(g, it.col, Math.trunc(lerp(it.p0[0], it.p1[0], tf)), Math.trunc(lerp(it.p0[1], it.p1[1], tf)), Math.max(2, 9 - i));
        }
        glow(g, [x, y], 24, it.col, 160);
        circle(g, [255, 255, 255], Math.trunc(x), Math.trunc(y), 7);
      } else if (it.k === "beam") {
        const a = 1 - Math.abs(2 * f - 1) * 0.6, w = Math.trunc(18 * a) + 4;
        line(g, it.col, it.p0, it.p1, w);
        line(g, [255, 255, 255], it.p0, it.p1, Math.max(2, Math.floor(w / 3)));
        glow(g, it.p1, 50, it.col, 160);
      } else if (it.k === "burst") {
        for (let i = 0; i < 14; i++) {
          const ang = i / 14 * Math.PI * 2 + 0.3, d = 12 + 60 * easeOut(f);
          circle(g, it.col, Math.trunc(it.p[0] + Math.cos(ang) * d), Math.trunc(it.p[1] + Math.sin(ang) * d), Math.max(1, Math.trunc(7 * (1 - f))));
        }
        circle(g, [255, 255, 255], Math.trunc(it.p[0]), Math.trunc(it.p[1]), Math.trunc(40 * (1 - f)) + 2, 3);
        glow(g, it.p, 70, it.col, Math.trunc(150 * (1 - f)));
      } else if (it.k === "shield") {
        const rad = AV_R + 14 + 10 * f;
        circle(g, it.col, Math.trunc(it.p[0]), Math.trunc(it.p[1]), Math.trunc(rad), 5);
        glow(g, it.p, Math.trunc(rad) + 20, it.col, Math.trunc(160 * (1 - f)));
      } else if (it.k === "ring") {
        const rad = 20 + 160 * easeOut(f);
        circle(g, it.col, Math.trunc(it.p[0]), Math.trunc(it.p[1]), Math.trunc(rad), Math.max(2, Math.trunc(8 * (1 - f))));
      }
    }
  }
  /** Animate one round.  Space / Enter jumps to the end; with waitClick it then waits for one more press. */
  async function playReveal(before, outcome, viewerId, waitClick) {
    const pos = layout(before, viewerId), acts = outcome.actions;
    const names = {};
    for (const p of before) names[p.id] = p.name;
    const [sched, death, end] = buildSchedule(pos, acts, outcome);
    const after = {};
    for (const p of outcome.players) after[p.id] = p;
    const lines = describeEvents(before, outcome);
    let tReal0 = now();
    const bannerT = end + 0.15, total = bannerT + 0.9;
    let logged = false, skip = false;
    while (true) {
      await frame();
      let t = (now() - tReal0) * speed();
      for (const e of await pumpEvents()) {
        if (isKey(e, "space", "return")) skip = true;
        if (e.type === "MOUSEBUTTONDOWN" && t > bannerT) skip = true;
      }
      if (skip && t < total) { tReal0 -= (total - t) / speed(); t = total; skip = false; }
      if (t >= bannerT && !logged) { ctx.log = ctx.log.concat([[`-- Round ${ctx.roundNo} --`, C_GOLD]], lines); logged = true; }
      const aliveVis = {};
      for (const p of before) aliveVis[p.id] = (p.id in death ? death[p.id] : 99) > t && p.alive;
      const bubbles = {};
      for (const pid of Object.keys(acts)) {
        const a = acts[pid];
        bubbles[pid] = [a.action, a.target !== null && a.target !== undefined ? names[a.target] : "", easeOut(t / 0.5)];
      }
      const bulletsVis = {};
      for (const pid of Object.keys(after)) bulletsVis[pid] = t >= 0.9 ? after[pid].bullets : before.find((p) => String(p.id) === pid).bullets;
      const shake = {};
      for (const pid of Object.keys(death)) if (death[pid] - t >= 0 && death[pid] - t < 0.3) shake[pid] = 6;
      drawHeader(ctx.roundNo, ctx.cfg.label);
      drawTable(before, pos, { aliveVis: aliveVis, bubbles: bubbles, tnow: t, shake: shake, bulletsVis: bulletsVis });
      drawSched(sched, t);
      drawLog();
      if (t >= bannerT) {
        const died = outcome.deaths.map((p) => names[p]);
        const a = Math.trunc(255 * Math.min(1, (t - bannerT) / 0.25));
        text("big", died.length ? elimText(died) : "Nobody died", [W / 2, 580], died.length ? C_RED : C_GREEN, "center", true, a);
      }
      if (t >= total) {
        if (waitClick) {
          text("body", "Click or press Space to continue", [W / 2, 640], C_GOLD, "midtop");
          if (OPTS.autoplay && t > total + 0.2) return;
          for (const e of await pumpEvents()) if (e.type === "MOUSEBUTTONDOWN" || isKey(e, "space", "return")) return;
          continue;
        }
        return;
      }
    }
  }

  // ============================================================== match, replay, game over
  async function runMatch() {
    const cfg = ctx.cfg, n = cfg.opponents + 1, hot = cfg.hotseat;
    let players = [];
    for (let i = 0; i < n; i++) players.push(hot ? C.makePlayer(i, `Player ${i + 1}`, true) : C.makePlayer(i, i === 0 ? "You" : `Bot ${i}`, i === 0));
    const bots = new Map();
    if (!hot) {
      for (const p of players.slice(1)) {
        const samples = n <= 3 ? null : Math.max(60, Math.floor(C.LEVELS[cfg.difficulty].samples * 3 / 4));
        bots.set(p.id, new C.ThinkingBot(p.id, cfg.difficulty, OPTS.seed === null ? new C.Rng() : new C.Rng(OPTS.seed + p.id),
          samples, ctx.store.load(p.id)));                       // this bot's saved memory
      }
    }
    ctx.log = [["Everyone starts with 0 bullets and one Reflect charge.", C_MUTED], ["Reload is risky: anyone who attacks you kills you.", C_MUTED]];
    ctx.insight = {};
    ctx.botNames = new Map(Array.from(bots.keys(), (pid) => [pid, players[pid].name]));
    let roundNo = 1;
    const history = [];
    ctx.inMatch = true;
    while (C.alivePlayers(players).length > 1) {
      ctx.roundNo = roundNo;
      const pub = players.map((p) => Object.assign({}, p));
      const actions = {};
      for (const [pid, b] of bots) if (players[pid].alive) actions[pid] = b.choose(pub, roundNo);
      if (bots.size) { ctx.nextInsight = {}; for (const [pid, b] of bots) if (pid in actions) ctx.nextInsight[pid] = b.last_probs.slice(); }
      for (const h of players.filter((p) => p.alive && p.human)) {
        if (hot) await passScreen(h.name, avatarColor(h));
        actions[h.id] = await humanTurn(h, players, roundNo, hot);
      }
      const outcome = C.resolveRound(players, actions);
      history.push(outcome.actions);
      await playReveal(players, outcome, hot ? Math.min.apply(null, players.filter((p) => p.alive).map((p) => p.id)) : 0, hot);
      if (bots.size) ctx.insight = ctx.nextInsight;
      for (const b of bots.values()) b.observe(pub, outcome.actions);
      players = outcome.players;
      roundNo++;
    }
    ctx.inMatch = false;
    ctx.botNames = new Map();
    return [players, roundNo - 1, history];
  }
  // wait for one press at the end of a replayed round
  async function waitNext() {
    text("body", "Click or press Space for the next round", [W / 2, 640], C_GOLD, "midtop");
    const t0 = now();
    while (true) {
      await frame();
      if (OPTS.autoplay && now() - t0 > 0.2) return;
      for (const e of await pumpEvents()) if (e.type === "MOUSEBUTTONDOWN" || isKey(e, "space", "return")) return;
    }
  }
  /** Replay a recorded game: one press per round (a press during the animation goes straight on). */
  async function replayGame(rec) {
    let players = [];
    for (let i = 0; i < rec.n; i++) players.push(C.makePlayer(i, i === 0 ? "You" : `Bot ${i}`, i === 0));
    const oldCfg = ctx.cfg, oldIn = ctx.inMatch;
    ctx.cfg = Object.assign({}, oldCfg, { label: `REPLAY of game #${rec.id}  |  Space = next round  |  Esc = stop` });
    ctx.inMatch = false; ctx.replaying = true;
    ctx.log = [[`Replay of game #${rec.id} (${rec.time}).`, C_GOLD]];
    ctx.insight = {}; ctx.botNames = new Map();
    try {
      for (let r = 0; r < rec.rounds.length; r++) {
        ctx.roundNo = r + 1;
        const outcome = C.resolveRound(players, C.roundActions(rec.rounds[r]));
        const end = buildSchedule(layout(players, 0), outcome.actions, outcome)[2];
        const full = (end + 0.15 + 0.9) / speed();
        const t0 = now();
        await playReveal(players, outcome, 0, false);
        if (now() - t0 >= full - 0.05) await waitNext();
        players = outcome.players;
      }
    } catch (err) { if (!(err instanceof ReplayStop)) throw err; }
    finally { ctx.cfg = oldCfg; ctx.inMatch = oldIn; ctx.replaying = false; }
  }
  // one-line summary of the recorded games
  function statsLine(stats) {
    if (!stats.games) return "No games recorded yet - your first one is about to be saved.";
    return `Games played: ${stats.games}   |   you won ${stats.wins}   bots won ${stats.losses}   draws ${stats.draws}`;
  }
  // result screen; returns 'replay', 'again', 'menu' or 'quit'
  async function gameOver(players, rounds, rec) {
    const live = C.alivePlayers(players);
    let msg, col;
    if (live.length === 1) {
      const w = live[0];
      msg = w.name !== "You" ? `${w.name} WON!` : "YOU WIN!";
      col = w.human ? C_GREEN : C_RED;
      if (!ctx.cfg.hotseat && !w.human) msg = `${w.name} wins - you lose`;
    } else { msg = "DRAW - nobody survives"; col = C_GOLD; }
    const pos = {};
    players.forEach((p, i) => { pos[p.id] = [W / 2 + (i - (players.length - 1) / 2) * 200, 150]; });
    const bReplay = new Rect(W / 2 - 410, 560, 190, 70), bAgain = new Rect(W / 2 - 200, 560, 190, 70);
    const bMenu = new Rect(W / 2 + 10, 560, 190, 70), bQuit = new Rect(W / 2 + 220, 560, 190, 70);
    const t0 = now();
    while (true) {
      await frame();
      const t = now() - t0;
      for (const e of await pumpEvents()) {
        if (e.type === "MOUSEBUTTONDOWN" && e.button === 1) {
          if (rec && bReplay.collidepoint(e.pos)) return "replay";
          if (bAgain.collidepoint(e.pos)) return "again";
          if (bMenu.collidepoint(e.pos)) return "menu";
          if (bQuit.collidepoint(e.pos)) return "quit";
        }
        if (isKey(e, "return")) return "again";
        if (isKey(e, "r") && rec) return "replay";
      }
      if (OPTS.autoplay && Q.get("hold") !== "gameover" && t > 0.8 / speed()) return "again";
      drawHeader(0, ctx.cfg.label);
      text("med", "MATCH OVER", [24, 22], C_GOLD);
      drawTable(players, pos, { tnow: t });
      const sc = easeOut(t / 0.6);
      glow(g, [W / 2, 380], Math.trunc(240 * sc) + 10, col, 70);
      text("title", msg, [W / 2, 380], col, "center", true, Math.trunc(255 * sc));
      text("body", `Match lasted ${rounds} round${rounds !== 1 ? "s" : ""}.`, [W / 2, 450], C_MUTED, "center");
      if (!ctx.cfg.hotseat) {
        text("small", rec ? `Saved as game #${rec.id} - the bots have learned from it.` : "This game could not be saved.", [W / 2, 482], C_MUTED, "center");
        text("small", statsLine(ctx.stats), [W / 2, 510], C_GOLD, "center");
      }
      for (const [r, label, c, show] of [[bReplay, "REPLAY (R)", C_PURPLE, !!rec], [bAgain, "AGAIN", C_GREEN, true],
        [bMenu, "MENU", C_BLUE, true], [bQuit, "QUIT", C_RED, true]]) {
        if (!show) continue;
        rrect(g, r, [28, 33, 68], 16); rrect(g, r, c, 16, 3);
        text("med", label, r.center, c, "center");
      }
    }
  }

  // ============================================================== tutorial (title page + Instruction page)
  const INSTRUCTION_COLS = [
    [["reload", "+1 bullet, defenseless"], ["defense", "free, defend bullets"], ["reflect", "only once, reflect bullets & kill attacker"]],
    [["shoot", "-1 bullet, attack another player"], ["bomb", "-(living players - 1) bullets, shoot everyone else"], ["mg", "-10 bullets, neglect defense and reflect"]],
  ];
  // hover demo of each action: [bullets per player, actions, caption]
  const DEMO_SCENES = {
    reload: [[0, 1], { 0: ["reload", null], 1: ["defense", null] }, "Reload: +1 bullet, but any attack would kill you"],
    shoot: [[1, 0], { 0: ["shoot", 1], 1: ["reload", null] }, "Hit!  Shoot beats Reload"],
    defense: [[0, 1], { 0: ["defense", null], 1: ["shoot", 0] }, "Blocked!  Defense beats Shoot"],
    bomb: [[2, 0, 0], { 0: ["bomb", null], 1: ["reload", null], 2: ["reload", null] }, "Bomb hits everyone else at once"],
    reflect: [[0, 1], { 0: ["reflect", null], 1: ["shoot", 0] }, "Reflected: the attacker dies"],
    mg: [[10, 1], { 0: ["mg", 1], 1: ["defense", null] }, "MG goes through Defense"],
  };
  // demo rounds already computed
  const DEMO_CACHE = {};
  // the demo round of one action, computed once with the real rules
  function demoRound(a) {
    if (!DEMO_CACHE[a]) {
      const [bullets, acts] = DEMO_SCENES[a];
      const players = bullets.map((b, i) => { const p = C.makePlayer(i, i === 0 ? "You" : `Bot ${i}`, i === 0); p.bullets = b; return p; });
      const actions = {};
      for (const pid of Object.keys(acts)) actions[pid] = { action: acts[pid][0], target: acts[pid][1] };
      DEMO_CACHE[a] = [players, C.resolveRound(players, actions)];
    }
    return DEMO_CACHE[a];
  }
  // the looping demo of one action in a popup panel
  function drawActionDemo(a, panel, t) {
    const [players, out] = demoRound(a);
    const n = players.length, cy = panel.y + 118;
    const pos = n === 2 ? { 0: [panel.x + 110, cy], 1: [panel.right - 110, cy] }
      : { 0: [panel.x + 110, cy - 8], 1: [panel.right - 110, cy - 56], 2: [panel.right - 110, cy + 36] };
    const [sched, death, end] = buildSchedule(pos, out.actions, out);
    t = t % (end + 1.3);
    rrect(g, panel, [14, 17, 38], 16);
    rrect(g, panel, ACT_COLOR[a], 16, 2);
    text("med", title(ACT_NAME[a]), [panel.centerx, panel.y + 12], ACT_COLOR[a], "midtop");
    const after = {};
    for (const p of out.players) after[p.id] = p;
    for (const p of players) {
      const [x, y] = pos[p.id];
      const alive = (p.id in death ? death[p.id] : 99) > t;
      const col = alive ? AVATAR_COLORS[p.id % 5] : [70, 72, 84], r = n === 3 ? 24 : 28;
      circle(g, col, Math.trunc(x), Math.trunc(y), r);
      circle(g, alive ? [230, 232, 240] : [110, 112, 124], Math.trunc(x), Math.trunc(y), r, 2);
      const b = t >= 0.9 ? after[p.id].bullets : p.bullets;
      text("tiny", `${p.name}  (${b})`, [x, y + r + 4], alive ? C_TEXT : C_MUTED, "midtop");
      if (t >= 0.15) {
        const k = out.actions[p.id].action;
        drawIcon(g, k, Math.trunc(x + (r + 34) * (p.id === 0 ? 1 : -1)), Math.trunc(y), 13, ACT_COLOR[k]);
      }
    }
    drawSched(sched, t);
    if (t >= end - 0.2) text("body", DEMO_SCENES[a][2], [panel.centerx, panel.bottom - 36], C_GOLD, "midtop");
  }
  // title page background
  const forest = new Image();
  forest.src = "assets/forest_bg.png";
  // the two introduction pages: [title, lines]
  const TUTORIAL = [["007", ["Multi-round simultaneous decision games, see who's the last to survive."]], ["Instruction", null]];

  // the two introduction pages (title page, Instruction page with hover demos)
  async function tutorial(startPage = 0) {
    let page = startPage;
    const bNext = new Rect(W / 2 + 20, H - 100, 260, 64), bBack = new Rect(W / 2 - 280, H - 100, 260, 64), bSkip = new Rect(W - 170, 20, 150, 44);
    let last = now(), demoA = null, demoT0 = 0;
    while (true) {
      await frame();
      for (const e of await pumpEvents()) {
        if (e.type === "MOUSEBUTTONDOWN" && e.button === 1) {
          if (bSkip.collidepoint(e.pos)) return;
          if (TUTORIAL[page][0] === "Instruction" && ctx.secretRect && ctx.secretRect.collidepoint(e.pos)) ctx.secretFound = true;
          if (bNext.collidepoint(e.pos)) { if (page === TUTORIAL.length - 1) return; page++; }
          if (bBack.collidepoint(e.pos) && page > 0) page--;
        }
        if (isKey(e, "return", "space", "right")) { if (page === TUTORIAL.length - 1) return; page++; }
      }
      if (OPTS.autoplay && now() - last > 0.25 / speed()) { last = now(); if (page === TUTORIAL.length - 1) return; page++; }
      const [ttl, lines] = TUTORIAL[page];
      if (page === 0 && forest.complete && forest.naturalWidth) g.drawImage(forest, 0, 0, W, H);   // forest behind the title page
      else fillScreen(C_BG);
      text(page === 0 ? "title" : "big", ttl, [W / 2, page ? 18 : 80], C_GOLD, "midtop", true);
      if (page) text("body", "Put your mouse on each action to see the animation", [W / 2, 76], C_MUTED, "midtop");
      ctx.secretRect = null;
      if (lines) {
        text("big", "How's your luck? Wanna Gamble?", [W / 2, 220], C_RED, "midtop", true);
        let y = 310;
        for (const ln of lines) for (const wl of wrap("body", ln, 980)) { text("body", wl, [W / 2, y], C_TEXT, "midtop", true); y += 32; }
      } else {
        const cards = {};
        INSTRUCTION_COLS.forEach((col, c) => col.forEach(([a, desc], i) => {
          const r = new Rect(50 + c * 600, 125 + i * 118, 580, 104);
          cards[a] = r;
          rrect(g, r, C_PANEL, 14); rrect(g, r, [60, 66, 110], 14, 2);
          drawIcon(g, a, r.x + 46, r.centery, 24, ACT_COLOR[a]);
          const key = new Rect(r.x + 92, r.y + 16, 34, 34);
          rrect(g, key, [28, 33, 68], 8); rrect(g, key, ACT_COLOR[a], 8, 2);
          text("body", ACT_KEY[a], key.center, ACT_COLOR[a], "center");
          text("med", title(ACT_NAME[a]), [key.right + 12, key.centery], ACT_COLOR[a], "midleft");
          wrap("body", desc, r.right - (r.x + 92) - 16).forEach((wl, j) => text("body", wl, [r.x + 92, r.y + 58 + j * 26], C_TEXT));
        }));
        text("body", "Note: bullet cancellation!      MG-MG: TIE!", [W / 2, 488], C_GOLD, "midtop");
        ctx.secretRect = new Rect(W / 2 - 300, 524, 600, 36);                 // hidden note: appears when clicked
        text("body", "Press here ->", [W / 2 - 150, 528], C_MUTED, "topright");
        if (ctx.secretFound) text("body", "Machine gun vs. Shoot: TIE!", [W / 2, 528], C_GOLD, "midtop");
        let hov = null;
        for (const a of Object.keys(cards)) if (cards[a].collidepoint(mouse)) { hov = a; break; }
        if (hov !== demoA) { demoA = hov; demoT0 = now(); }
        if (demoA) {
          const r = cards[demoA], panel = new Rect(0, 0, 520, 250);
          panel.y = Math.min(Math.max(r.y - 20, 100), H - 120 - 250);
          panel.x = r.x < W / 2 ? 50 + 600 + 30 : 50 + 30;                    // over the other column
          drawActionDemo(demoA, panel, now() - demoT0);
        }
      }
      text("tiny", `${page + 1} / ${TUTORIAL.length}`, [W / 2, H - 24], C_MUTED, "midtop");
      for (const [r, label, c, show] of [[bNext, page === 0 ? "Instruction" : "Next", C_GOLD, true], [bBack, "BACK", C_BLUE, page > 0], [bSkip, "SKIP", C_MUTED, true]]) {
        if (!show) continue;
        rrect(g, r, [28, 33, 68], 16); rrect(g, r, c, 16, 3);
        text("med", label, r.center, c, "center");
      }
    }
  }

  // ============================================================== game setup
  async function menu(cfg) {
    cfg.opponents = 4;                                         // Game Setup always opens on 1 vs 4 Bots
    const cards = [[1, "1 vs 1 Bot", "1H1B - a duel against the solved-game bot"], [2, "1 vs 2 Bots", "1H2B - three-way survival"],
      [4, "1 vs 4 Bots", "1H4B - five-player free-for-all"]];
    const cardRects = [0, 1, 2].map((i) => new Rect(70 + i * 380, 130, 360, 150));
    const diffs = ["easy", "normal", "hard"], diffRects = [0, 1, 2].map((i) => new Rect(70 + i * 170, 360, 150, 54));
    const timers = [3, 5, 10], timerRects = [0, 1, 2].map((i) => new Rect(720 + i * 130, 360, 110, 54));
    const hotRect = new Rect(70, 480, 230, 56), startRect = new Rect(W / 2 - 140, 600, 280, 76), helpRect = new Rect(70, 610, 200, 56);
    const diffDesc = { easy: "Makes deliberate mistakes. A good first opponent.", normal: "Plays near-optimal and slowly learns your habits.",
      hard: "Near-optimal and ruthlessly exploits your habits." };
    const t0 = now();
    while (true) {
      await frame();
      for (const e of await pumpEvents()) {
        if (e.type === "MOUSEBUTTONDOWN" && e.button === 1) {
          cardRects.forEach((r, i) => { if (r.collidepoint(e.pos)) cfg.opponents = cards[i][0]; });
          diffRects.forEach((r, i) => { if (r.collidepoint(e.pos)) cfg.difficulty = diffs[i]; });
          timerRects.forEach((r, i) => { if (r.collidepoint(e.pos)) cfg.timer = timers[i]; });
          if (hotRect.collidepoint(e.pos)) cfg.hotseat = !cfg.hotseat;
          if (helpRect.collidepoint(e.pos)) await tutorial();
          if (startRect.collidepoint(e.pos)) return cfg;
        }
        if (isKey(e, "return")) return cfg;
      }
      if (OPTS.autoplay && now() - t0 > 0.3 / speed()) { cfg.opponents = OPTS.mode; return cfg; }
      fillScreen(C_BG);
      text("big", "GAME SETUP", [W / 2, 40], C_GOLD, "midtop", true);
      cardRects.forEach((r, ci) => {
        const [n, name, desc] = cards[ci], sel = cfg.opponents === n;
        rrect(g, r, sel ? C_PANEL2 : C_PANEL, 18);
        rrect(g, r, sel ? C_GOLD : [60, 66, 110], 18, sel ? 4 : 2);
        for (let i = 0; i <= n; i++) circle(g, i ? AVATAR_COLORS[i % 5] : [70, 175, 255], Math.trunc(r.centerx + (i - n / 2) * 46), r.y + 44, 17);
        text("med", name, [r.centerx, r.y + 78], C_TEXT, "midtop");
        wrap("tiny", desc, r.w - 30).forEach((ln, j) => text("tiny", ln, [r.centerx, r.y + 112 + j * 18], C_MUTED, "midtop"));
      });
      text("body", "Bot difficulty", [70, 322], C_MUTED);
      diffRects.forEach((r, i) => {
        const sel = cfg.difficulty === diffs[i];
        rrect(g, r, sel ? C_PANEL2 : C_PANEL, 14); rrect(g, r, sel ? C_GREEN : [60, 66, 110], 14, sel ? 3 : 2);
        text("body", diffs[i].toUpperCase(), r.center, C_TEXT, "center");
      });
      text("small", diffDesc[cfg.difficulty], [70, 428], C_MUTED);
      text("body", "Round timer", [720, 322], C_MUTED);
      timerRects.forEach((r, i) => {
        const sel = cfg.timer === timers[i];
        rrect(g, r, sel ? C_PANEL2 : C_PANEL, 14); rrect(g, r, sel ? C_GREEN : [60, 66, 110], 14, sel ? 3 : 2);
        text("body", `${timers[i]} s`, r.center, C_TEXT, "center");
      });
      const on = cfg.hotseat;
      rrect(g, hotRect, on ? [90, 36, 46] : C_PANEL, 14); rrect(g, hotRect, on ? C_RED : [60, 66, 110], 14, 3);
      text("body", "Testing mode", hotRect.center, C_TEXT, "center");                 // red = ON
      text("bodyb", "press SPACE to skip the animation of the round", [hotRect.right + 28, hotRect.centery], C_TEXT, "midleft");
      text("body", statsLine(ctx.stats), [W / 2, 552], C_GOLD, "midtop");
      rrect(g, helpRect, C_PANEL, 14); rrect(g, helpRect, C_BLUE, 14, 2);
      text("body", "HOW TO PLAY", helpRect.center, C_BLUE, "center");
      rrect(g, startRect, [40, 60, 40], 18); rrect(g, startRect, C_GREEN, 18, 4);
      text("big", "START", startRect.center, C_GREEN, "center");
    }
  }

  // ============================================================== main loop
  let retuneDone = false;
  // run the after-game re-tuning shortly after the game-over screen appears
  function retuneSoon() {
    // the re-tuning runs a moment after the game-over screen appears (it re-scores the visitor's games)
    setTimeout(() => {
      try {
        const t0 = now();
        const p = C.retuneAfterGame(ctx.store);
        retuneDone = true;
        document.body.dataset.retune = `ok games=${ctx.store.records().length} w=${p.STREAK_PRIOR_WEIGHT} r=${p.PATTERN_ROUNDS} ms=${Math.round((now() - t0) * 1000)}`;
      } catch (err) { console.warn("[007] parameter update skipped:", err); document.body.dataset.retune = "error: " + err.message; }
    }, 400);
  }
  // the game: tutorial -> setup -> match -> game over, in a loop
  async function main() {
    let backend = null;
    if (OPTS.storage !== "memory") { try { localStorage.setItem("007-test", "1"); localStorage.removeItem("007-test"); backend = localStorage; } catch (e) { backend = null; } }
    ctx.store = new C.BotStore(backend || new C.MemoryBackend());
    C.applyParams(ctx.store.loadParams());
    ctx.stats = C.BotStore.summary(ctx.store.records());
    let cfg = { opponents: OPTS.mode, difficulty: OPTS.difficulty, timer: OPTS.timer, hotseat: false };
    let games = 0;
    if (!OPTS.skipTutorial) await tutorial(Math.max(0, Math.min(1, Number(Q.get("page") || 1) - 1)));
    while (true) {
      cfg = await menu(cfg);
      while (true) {
        ctx.cfg = Object.assign({}, cfg);
        ctx.cfg.label = `${cfg.hotseat ? "Testing mode" : `1 vs ${cfg.opponents} Bot${cfg.opponents > 1 ? "s" : ""}`}  |  ${title(cfg.difficulty)}  |  ${cfg.timer}s`
          + (cfg.hotseat ? "" : `  |  game #${ctx.stats.games + 1}`) + "  |  Space = skip";
        if (retuneDone) { retuneDone = false; C.applyParams(ctx.store.loadParams()); }
        let res;
        try { res = await runMatch(); }
        catch (err) { if (err instanceof ReturnToMenu) { ctx.inMatch = false; ctx.botNames = new Map(); break; } throw err; }
        const [players, rounds, history] = res;
        games++;
        let rec = null;
        if (!cfg.hotseat) {
          const live = C.alivePlayers(players);
          rec = C.makeRecord(ctx.store.nextId(), cfg, players.length, history, live.length === 1 ? live[0].id : null);
          try { ctx.store.addGame(rec); } catch (err) { console.warn("[007] could not save the game:", err); rec = null; }
          retuneSoon();
          ctx.stats = C.BotStore.summary(ctx.store.records());
        }
        if (OPTS.games && games >= OPTS.games) { window.__done = true; return; }
        let choice = await gameOver(players, rounds, rec);
        while (choice === "replay") { await replayGame(rec); choice = await gameOver(players, rounds, rec); }
        if (choice === "quit") { await tutorial(); break; }
        if (choice === "menu") break;
      }
    }
  }
  window.Game007 = { ctx: ctx, OPTS: OPTS };
  main().catch((err) => { console.error(err); document.body.dataset.error = err.message; fillScreen(C_BG); text("body", "Something went wrong: " + err.message, [40, 40], C_RED); });
})();
