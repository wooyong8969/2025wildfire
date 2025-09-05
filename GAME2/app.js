// ---------- Config ----------
const N = 10;
const M = 10;

const CLUSTER = { minCount: 1, maxCount: 2, sigma: 1.5, base: 0.02, strength: 0.95 };
const WIND = { minSpeed: 1, maxSpeed: 10 };
const DYNAMICS = {
  stay: 0.75,
  spreadBase: 0.05,
  spark: 0.003,
  windGain: 3.0,
  radius: 2
};

let gridState = [];
let waterPlaced = new Set();
let waterRemaining = 5;
let score = 0;
let round = 1;
let currentWind = { dir: 0, speed: 0 };
let gameOver = false;

function key(r, c) { return `${r},${c}`; }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function randInt(a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; }

function clusteredGrid(rows, cols, opts = CLUSTER) {
  const k = randInt(opts.minCount, opts.maxCount);
  const centers = Array.from({ length: k }, () => ({ r: Math.random() * rows, c: Math.random() * cols }));
  const out = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      let p = opts.base;
      for (const { r: cr, c: cc } of centers) {
        const d2 = (r - cr) ** 2 + (c - cc) ** 2;
        const gauss = Math.exp(-d2 / (2 * opts.sigma ** 2));
        p = Math.max(p, opts.base + opts.strength * gauss);
      }
      row.push(Math.random() < p);
    }
    out.push(row);
  }
  let fireCount = out.flat().filter(Boolean).length;
  let attempts = 0;
  while (fireCount < 3 && attempts++ < 50) {
    const r = randInt(0, rows - 1), c = randInt(0, cols - 1);
    if (!out[r][c]) { out[r][c] = true; fireCount++; }
  }
  return out;
}

function rollWind() {
  const dir = Math.random() * 360;
  const speed = randInt(WIND.minSpeed, WIND.maxSpeed);
  currentWind = { dir, speed };
  renderWind();
}

function neighbors(r, c, rows = N, cols = M, radius = DYNAMICS.radius) {
  const list = [];
  for (let dr = -radius; dr <= radius; dr++) {
    for (let dc = -radius; dc <= radius; dc++) {
      if (dr === 0 && dc === 0) continue;
      const rr = r + dr, cc = c + dc;
      if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
        list.push([rr, cc, dr, dc]);
      }
    }
  }
  return list;
}

function generateNextGrid(current) {
  const theta = (currentWind.dir * Math.PI) / 180;
  const windVec = { r: Math.sin(theta), c: Math.cos(theta) };
  const rows = current.length, cols = current[0].length;
  const next = Array.from({ length: rows }, () => Array(cols).fill(false));
  const fireIntent = {}; // 🔥 불 번질 의도 기록

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const k = key(r, c);
      if (waterPlaced.has(k)) {
        fireIntent[k] = false;
      }

      const hereBurns = current[r][c] === true;
      if (hereBurns && Math.random() < DYNAMICS.stay) {
        next[r][c] = true;
        continue;
      }

      let survive = 1.0;
      for (const [rr, cc, dr, dc] of neighbors(r, c, rows, cols)) {
        if (!current[rr][cc]) continue;
        const dist = Math.hypot(dr, dc);
        const dot = windVec.r * (-dr) + windVec.c * (-dc);
        const align = dot / (dist + 1e-6);
        const angleBias = Math.max(0, align);
        const windEffect = 1 + DYNAMICS.windGain * angleBias * currentWind.speed;
        const p_i = clamp((DYNAMICS.spreadBase * windEffect) / dist, 0, 0.95);
        survive *= (1 - p_i);
      }

      let p = 1 - survive;
      p = 1 - (1 - p) * (1 - DYNAMICS.spark);
      const willBurn = Math.random() < p;

      if (waterPlaced.has(k)) {
        fireIntent[k] = willBurn;
        next[r][c] = false;
      } else {
        next[r][c] = willBurn;
      }
    }
  }

  return { next, fireIntent };
}


// ---------- DOM ----------
const gridEl = document.getElementById('grid');
const scoreEl = document.getElementById('score');
const roundEl = document.getElementById('round');
const deltaEl = document.getElementById('delta');
const nextBtn = document.getElementById('nextBtn');
const resetBtn = document.getElementById('resetBtn');

const windDirEl = document.getElementById('windDir');
const windSpeedEl = document.getElementById('windSpeed');
const windArrowEl = document.getElementById('windArrow');

function renderGrid() {
  gridEl.style.setProperty('--cols', M);
  gridEl.style.setProperty('--rows', N);
  gridEl.innerHTML = '';

  for (let r = 0; r < N; r++) {
    for (let c = 0; c < M; c++) {
      const cell = document.createElement('button');
      cell.className = 'cell';
      cell.setAttribute('role', 'gridcell');
      cell.setAttribute('aria-label', `${r+1}행 ${c+1}열`);
      cell.dataset.r = r;
      cell.dataset.c = c;

      const burning = gridState[r][c] === true;
      const k = key(r, c);
      const hasWater = waterPlaced.has(k);

      if (burning) cell.classList.add('onfire');
      if (hasWater) cell.classList.add('water');
      if (!burning && !hasWater) cell.classList.add('normal');

      cell.addEventListener('click', () => toggleWater(r, c));
      cell.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleWater(r, c);
        }
      });

      gridEl.appendChild(cell);
    }
  }
}

function toggleWater(r, c) {
  const k = key(r, c);
  if (waterPlaced.has(k)) {
    waterPlaced.delete(k);
    waterRemaining++;
  } else if (waterRemaining > 0) {
    waterPlaced.add(k);
    waterRemaining--;
  }
  renderGrid();
  updateWaterDisplay();
}

function updateScoreboard(delta=0) {
  scoreEl.textContent = String(score);
  roundEl.textContent = String(round);
  // deltaEl이 있으면 업데이트 (없어도 오류 방지)
  if (deltaEl) {
    deltaEl.textContent = delta > 0 ? `+${delta}` : String(delta);
  }
  
  // 디버깅: 점수 업데이트 확인
  console.log('점수 업데이트:', { score, round, delta });
}

function renderWind() {
  windDirEl.textContent = currentWind.dir.toFixed(0);
  windSpeedEl.textContent = String(currentWind.speed);
  // SVG는 위쪽(북) 방향이 -90° 기준이 아님. 현재 라인은 아래->위로 그려져 있으므로 0°=동을 맞추기 위해 보정
  // our base arrow points up (from y=80 to y=20). Up is 270° in our definition.
  // We want 0° (east) to be displayed when dir=0 -> rotate to 0° visual.
  // Base arrow points up (270°), so rotate by (dir - 270)
  const rot = currentWind.dir - 270;
  windArrowEl.style.transform = `rotate(${rot}deg)`;
}

// 불이 난 칸 수 계산
function countFires(grid) {
  let count = 0;
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      if (grid[r][c]) count++;
    }
  }
  return count;
}

// 물 표시 업데이트
function updateWaterDisplay() {
  const waterEl = document.getElementById('waterRemaining');
  if (waterEl) {
    waterEl.textContent = String(waterRemaining);
  }
}

// 게임 오버 화면 표시
function showGameOver() {
  // 게임 오버 메시지를 alert로 표시하고 자동으로 리셋
  alert(`게임 오버!\n최종 점수: ${score}점\n${round}라운드까지 생존`);
  
  // 자동으로 게임 리셋
  resetAll();
}

// 평가: 물이 있는 칸에 불이 번지면 불 끄고 점수 획득
function evaluateAndAdvance() {
  rollWind();
  const { next, fireIntent } = generateNextGrid(gridState);

  console.log('현재 격자:', gridState);
  console.log('다음 격자:', next);
  console.log('물 배치 위치:', Array.from(waterPlaced));

  let correctPredictions = 0;
  for (const k of waterPlaced) {
    if (fireIntent[k]) {
      correctPredictions++;
      console.log(`🔥 예측 성공! ${k}에 불이 번지려다 물로 막음`);
    }
  }

  score += correctPredictions;
  updateScoreboard(correctPredictions);

  const fireCount = countFires(next);
  if (fireCount >= 60) {
    gameOver = true;
    showGameOver();
    return;
  }

  gridState = next;
  round += 1;
  waterPlaced.clear();
  waterRemaining = 5;
  renderGrid();
  updateWaterDisplay();
}
  

function randomizeCurrentGrid() {
  // 테스트용 격자 생성 - 확산 가능한 불 배치
  gridState = [
    [true, true, true, false, false, false, false, false, false, false],
    [true, true, true, false, false, false, false, false, false, false],
    [true, true, true, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false],
    [false, false, false, false, false, false, false, false, false, false]
  ];
  
  waterPlaced.clear();
  waterRemaining = 5;
  
  // 디버깅: 생성된 격자 상태 확인
  const fireCount = countFires(gridState);
  console.log('테스트 격자 생성됨:', { fireCount, gridState });
  
  updateScoreboard(0);
  renderGrid();
  updateWaterDisplay();
}

function resetAll() {
  score = 0;
  round = 1;
  gameOver = false;
  waterPlaced.clear();
  waterRemaining = 5;
  randomizeCurrentGrid();
  rollWind();
  updateWaterDisplay();
}

// ---------- Init ----------
function init() {
  // DOM 요소들이 모두 로드된 후 초기화
  setTimeout(() => {
    randomizeCurrentGrid();
    rollWind();
    updateWaterDisplay();
    
    nextBtn.addEventListener('click', evaluateAndAdvance);
    resetBtn.addEventListener('click', resetAll);
  }, 100);
}

document.addEventListener('DOMContentLoaded', init);
