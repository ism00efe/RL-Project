// Plays recording format v1 (docs/RECORDING.md). Knows nothing about MuJoCo or genomes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
$("stage").appendChild(renderer.domElement);

// Scene colors come from the page's CSS tokens so the floor follows the light/dark theme.
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const scene = new THREE.Scene();
scene.background = new THREE.Color();
scene.fog = new THREE.Fog(0, 6, 22);

const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 100);
camera.position.set(1.6, 0.9, 1.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.4;
controls.maxDistance = 12;

scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b0a4, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.radius = 4;
sun.shadow.bias = -0.0004;
Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.1, far: 12 });
scene.add(sun, sun.target);

const checker = document.createElement("canvas");
checker.width = checker.height = 256;
function paintChecker() {
  const g = checker.getContext("2d");
  g.fillStyle = css("--tile-a"); g.fillRect(0, 0, 256, 256);
  g.fillStyle = css("--tile-b"); g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
}
function checkerTexture() {
  paintChecker();
  const t = new THREE.CanvasTexture(checker);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(100, 100); // 0.5 m tiles on a 100 m plane
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const ground = new THREE.Mesh(new THREE.PlaneGeometry(100, 100),
  new THREE.MeshStandardMaterial({ map: checkerTexture(), roughness: 0.95 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

function applyTheme() {
  scene.background.setStyle(css("--bg"), THREE.SRGBColorSpace);
  scene.fog.color.copy(scene.background);
  paintChecker();
  ground.material.map.needsUpdate = true;
}
applyTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

// MuJoCo is z-up; three is y-up. Everything from the recording lives under `world`.
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);

// ---------- recording -> meshes ----------
function geomMesh(g) {
  let geo;
  const s = g.size;
  if (g.type === "box") geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  else if (g.type === "sphere") geo = new THREE.SphereGeometry(s[0], 24, 16);
  else if (g.type === "ellipsoid") { geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s[0], s[1], s[2]); }
  else if (g.type === "capsule" || g.type === "cylinder") {
    geo = g.type === "capsule" ? new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16)
                               : new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 20);
    geo.rotateX(Math.PI / 2); // three: along y; MuJoCo: along local z
  } else return null;
  const [r, gr, b] = g.rgba;
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(r, gr, b, THREE.SRGBColorSpace), roughness: 0.55, metalness: 0.05 }));
  mesh.position.fromArray(g.pos);
  mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); // wxyz -> xyzw
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

// Playback starts paused for viewers who ask for reduced motion.
let rec = null, parts = [], t = 0, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
const tmpQ0 = new THREE.Quaternion(), tmpQ1 = new THREE.Quaternion();
const tmpV0 = new THREE.Vector3(), tmpV1 = new THREE.Vector3();

function load(r) {
  if (r.format !== "evo-recording" || r.version !== 1) throw new Error(`unsupported recording ${r.format} v${r.version}`);
  parts.forEach((p) => world.remove(p));
  parts = r.parts.map((p) => {
    const obj = new THREE.Group();
    p.geoms.map(geomMesh).filter(Boolean).forEach((m) => obj.add(m));
    world.add(obj);
    return obj;
  });
  rec = r;
  t = 0;
  const m = r.meta;
  $("genome").textContent = m.genome_id;
  $("generation").textContent = m.generation;
  $("parent").textContent = m.parent_id ?? "—";
  $("env").textContent = m.env;
  $("fitness-name").textContent = m.fitness_name === "mean_vx" ? "mean speed" : (m.fitness_name || "fitness");
  $("fitness").textContent = typeof m.fitness === "number"
    ? `${m.fitness.toFixed(2)}${m.fitness_name === "mean_vx" ? " m/s" : ""}` : m.fitness;
  $("parts").textContent = r.parts.length;
  $("error").hidden = true;
  pose(0);
  // Start the camera behind-left of the root, looking at it.
  const root = rootWorld(new THREE.Vector3());
  camera.position.copy(root).add(new THREE.Vector3(1.4, 0.8, 1.6));
  controls.target.copy(root);
}

function duration() { return rec ? (rec.num_frames - 1) * rec.dt : 0; }

function pose(time) {
  const f = Math.min(Math.max(time / rec.dt, 0), rec.num_frames - 1);
  const i = Math.floor(f), j = Math.min(i + 1, rec.num_frames - 1), a = f - i;
  const P = rec.frames.pos, Q = rec.frames.quat;
  parts.forEach((obj, k) => {
    obj.position.copy(tmpV0.fromArray(P[i][k]).lerp(tmpV1.fromArray(P[j][k]), a));
    const q0 = Q[i][k], q1 = Q[j][k];
    tmpQ0.set(q0[1], q0[2], q0[3], q0[0]);
    tmpQ1.set(q1[1], q1[2], q1[3], q1[0]);
    obj.quaternion.copy(tmpQ0.slerp(tmpQ1, a));
  });
}

function rootWorld(out) {
  return parts.length ? parts[0].getWorldPosition(out) : out.set(0, 0, 0);
}

// ---------- follow cam + loop ----------
const clock = new THREE.Clock();
const prevRoot = new THREE.Vector3(), curRoot = new THREE.Vector3();

function tick() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (rec) {
    if (playing) {
      t += dt * Number($("speed").value);
      if (t > duration()) t = 0;
      $("scrub").value = String(t / duration());
    }
    rootWorld(prevRoot);
    pose(t);
    rootWorld(curRoot);
    // Follow: shift camera and target by the root's motion (orbit offset stays the user's).
    const d = curRoot.clone().sub(prevRoot);
    if (d.length() < 1) { camera.position.add(d); controls.target.add(d); }
    else { camera.position.add(curRoot.clone().sub(controls.target)); controls.target.copy(curRoot); }
    sun.position.copy(curRoot).add(new THREE.Vector3(2.5, 5, 1.5));
    sun.target.position.copy(curRoot);
    $("time").textContent = `${t.toFixed(1)} s`;
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------- UI ----------
$("play").textContent = playing ? "Pause" : "Play";
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; };
$("scrub").oninput = (e) => { if (rec) { t = Number(e.target.value) * duration(); } };
window.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); $("play").click(); } });

function showError(msg) { const el = $("error"); el.textContent = msg; el.hidden = false; }

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function select(file) {
  try {
    load(await fetchJSON(`recordings/${file}`));
    const u = new URL(location.href);
    u.searchParams.set("r", file);
    history.replaceState(null, "", u);
  } catch (e) { showError(String(e)); }
}

(async () => {
  try {
    const index = await fetchJSON("recordings/index.json");
    for (const item of index) $("pick").add(new Option(item.label, item.file));
    const wanted = new URL(location.href).searchParams.get("r");
    const first = index.find((x) => x.file === wanted) ?? index[0];
    $("pick").value = first.file;
    $("pick").onchange = (e) => select(e.target.value);
    await select(first.file);
  } catch (e) { showError(`Could not load recordings: ${e.message}`); }
  tick();
})();
