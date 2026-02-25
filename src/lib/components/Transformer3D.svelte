<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import type { TransformerStep } from '$lib/types';

	/** All transformer steps (one per layer) — or raw battle matrices */
	export let steps: TransformerStep[] = [];
	/** Raw layer objects from battle_data (may include connections[]) or plain number[][] matrices */
	export let battleLayers: any[] = [];
	/** Current token index — controls which column/row highlights */
	export let currentStep: number = 0;

	let container: HTMLDivElement;
	let renderer: any = null;
	let scene: any = null;
	let camera: any = null;
	let animFrame: number;
	let THREE: any = null;
	let planeMeshes: any[] = [];
	let planeMaterials: any[] = [];
	let resizeObserver: ResizeObserver | null = null;
	let synapseMeshes: any[] = [];

	// Canvas texture refs (one per layer)
	let textures: any[] = [];
	let noiseOffset = 0;
	let lastTextureStep = -1;

	// Extract 2D attention matrices and connections from battleLayers (handles both shapes)
	$: layerObjects = battleLayers.length > 0 ? battleLayers : [];

	$: matrices = layerObjects.length > 0
		? layerObjects.map((l: any) => {
			// Layer object with attention_matrix field
			if (l && l.attention_matrix) return l.attention_matrix;
			// Layer object — rebuild matrix from connections if available
			if (l && l.connections) {
				const nT = getMaxToken(l.connections) + 1;
				const mat: number[][] = Array.from({length: nT}, () => new Array(nT).fill(0));
				l.connections.forEach((c: any) => {
					if (c.source < nT && c.target < nT) mat[c.target][c.source] = c.weight;
				});
				return mat;
			}
			// Plain number[][] passed directly
			if (Array.isArray(l) && Array.isArray(l[0])) return l;
			return null;
		}).filter(Boolean)
		: steps.map(s => s.attention_matrix);

	$: connections = layerObjects.map((l: any) => {
		if (l && l.connections) return l.connections as {source: number, target: number, weight: number}[];
		return [];
	});

	$: nLayers = matrices.length || 4;

	function getMaxToken(conns: {source: number, target: number, weight: number}[]): number {
		let max = 0;
		conns.forEach(c => { max = Math.max(max, c.source, c.target); });
		return max;
	}

	function buildTexture(matrix: number[][], tokenIdx: number): HTMLCanvasElement {
		const size = 256;
		const canvas = document.createElement('canvas');
		canvas.width = size;
		canvas.height = size;
		const ctx = canvas.getContext('2d')!;

		const T = matrix.length;
		if (T === 0) return canvas;
		const cellW = size / T;
		const cellH = size / T;

		// Near-black dark red background
		ctx.fillStyle = '#0a0002';
		ctx.fillRect(0, 0, size, size);

		for (let r = 0; r < T; r++) {
			for (let c = 0; c < T; c++) {
				const norm = Math.max(Math.min(matrix[r][c], 1.0), 0);

				if (norm < 0.02) continue;

				// Dark-red → crimson → hot-orange gradient matching "overloaded/danger" theme
				const red   = Math.round(26  + 229 * norm);   // 26  → 255
				const green = Math.round(0   + 102 * norm);   // 0   → 102
				const blue  = Math.round(5   + 34  * norm);   // 5   → 34
				const alpha = 0.35 + 0.65 * norm;
				ctx.fillStyle = `rgba(${red},${green},${blue},${alpha})`;
				ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
			}
		}

		// Highlight current token row/column with cyan (intentional contrast: signal against dense noise)
		if (tokenIdx >= 0 && tokenIdx < T) {
			ctx.strokeStyle = 'rgba(0, 230, 255, 0.55)';
			ctx.lineWidth = 2;
			ctx.shadowColor = 'rgba(0, 200, 255, 0.4)';
			ctx.shadowBlur = 6;
			ctx.strokeRect(0, tokenIdx * cellH, size, cellH);
			ctx.strokeRect(tokenIdx * cellW, 0, cellW, size);
			ctx.shadowBlur = 0;
		}

		return canvas;
	}

	function updateTextures() {
		if (!THREE || !scene || matrices.length === 0) return;
		if (lastTextureStep === currentStep) return; // no change, skip rebuild
		lastTextureStep = currentStep;

		matrices.forEach((matrix, i) => {
			if (!planeMeshes[i] || !matrix) return;
			const canvas = buildTexture(matrix, currentStep);
			if (textures[i]) {
				textures[i].image = canvas;
				textures[i].needsUpdate = true;
			}
		});
	}

	function getTokenWorldPos(tokenIdx: number, nTokens: number, planeMesh: any, planeSize: number) {
		// Grid layout: token idx mapped to (col, row) in a roughly-square grid
		const gridW = Math.ceil(Math.sqrt(nTokens));
		const col = tokenIdx % gridW;
		const row = Math.floor(tokenIdx / gridW);
		const localX = (col / Math.max(gridW - 1, 1) - 0.5) * planeSize * 0.85;
		const localY = (row / Math.max(gridW - 1, 1) - 0.5) * planeSize * 0.85;

		// Transform local plane coords to world space
		const pos = new THREE.Vector3(localX, localY, 0);
		pos.applyMatrix4(planeMesh.matrixWorld);
		return pos;
	}

	function buildSynapses(
		matrixA: number[][],
		planeA: any,
		planeB: any,
		nTokens: number,
		planeSize: number
	): any {
		const positions: number[] = [];
		const colors: number[] = [];

		// Full dense all-to-all connections using attention matrix weights
		// Mirror lower-triangular matrix to get symmetric weights for full visual density
		for (let s = 0; s < nTokens; s++) {
			for (let t = 0; t < nTokens; t++) {
				const wLower = matrixA[s] ? (matrixA[s][t] ?? 0) : 0;
				const wUpper = matrixA[t] ? (matrixA[t][s] ?? 0) : 0;
				const w = Math.max(wLower, wUpper);
				if (w < 0.01) continue; // near-zero threshold for true density

				const posA = getTokenWorldPos(s, nTokens, planeA, planeSize);
				const posB = getTokenWorldPos(t, nTokens, planeB, planeSize);

				positions.push(posA.x, posA.y, posA.z);
				positions.push(posB.x, posB.y, posB.z);

				// Weight-based gradient:
				// w ~0.0 → near-invisible dark maroon (0.04, 0, 0)
				// w ~0.3 → dim crimson (0.35, 0.02, 0)
				// w ~0.6 → saturated red-orange (0.7, 0.1, 0)
				// w ~1.0 → bright hot orange (1.0, 0.4, 0)
				// Using cubic ease so low weights are very dim and high weights pop
				const wCurved = w * w * (3 - 2 * w); // smoothstep
				const r = 0.04 + 0.96 * wCurved;
				const g = 0.00 + 0.40 * wCurved * wCurved; // green only appears at high weights
				const b = 0.0;
				colors.push(r, g, b, r, g, b);
			}
		}

		if (positions.length === 0) return null;

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
		geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

		const material = new THREE.LineBasicMaterial({
			vertexColors: true,
			transparent: true,
			opacity: 0.9,  // high — vertex color brightness controls perceived weight, not this
			blending: THREE.AdditiveBlending,
			depthWrite: false,
		});

		return new THREE.LineSegments(geometry, material);
	}

	function disposeSynapses() {
		synapseMeshes.forEach(m => {
			if (!m) return;
			m.geometry?.dispose();
			m.material?.dispose();
			scene?.remove(m);
		});
		synapseMeshes = [];
	}

	function rebuildSynapses() {
		if (!THREE || !scene || planeMeshes.length < 2) return;
		disposeSynapses();

		// Ensure world matrices are up to date before computing positions
		scene.updateMatrixWorld(true);

		const nTokens = matrices.length > 0 ? matrices[0].length : 14;
		const planeSize = 2.8;

		for (let i = 0; i < planeMeshes.length - 1; i++) {
			const matA = matrices[i] || [];
			const mesh = buildSynapses(matA, planeMeshes[i], planeMeshes[i + 1], nTokens, planeSize);
			if (mesh) {
				scene.add(mesh);
				synapseMeshes.push(mesh);
			}
		}
	}

	async function initScene() {
		if (!browser || !container) return;

		THREE = await import('three');

		scene = new THREE.Scene();
		scene.background = new THREE.Color(0x080a0e);

		// Camera
		const w = container.clientWidth || 400;
		const h = container.clientHeight || 400;
		camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
		// Front-left elevated — plates recede into depth along Z, visible as staircase
		camera.position.set(-4.5, 2.5, 6.0);
		camera.lookAt(0, 0, 0);

		// Renderer
		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setSize(w, h);
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		container.appendChild(renderer.domElement);

		// Lighting — dark theme with red danger tint
		const ambient = new THREE.AmbientLight(0x1a0a0a, 0.3);
		scene.add(ambient);
		const dirLight = new THREE.DirectionalLight(0xcc4422, 0.35);
		dirLight.position.set(3, 5, 3);
		scene.add(dirLight);
		// Red rim light for plane edge tinting
		const rimLight = new THREE.PointLight(0x550000, 0.15);
		rimLight.position.set(0, 0, 0);
		scene.add(rimLight);

		// Create glass planes stacked along Z axis (receding into depth) with slight tilt
		const planeSize = 2.8;
		const zSpacing = 2.0;   // depth spacing between layers
		const layers = nLayers;
		const totalDepth = (layers - 1) * zSpacing;

		for (let i = 0; i < layers; i++) {
			const canvas = document.createElement('canvas');
			canvas.width = 256;
			canvas.height = 256;

			const texture = new THREE.CanvasTexture(canvas);
			texture.minFilter = THREE.LinearFilter;
			texture.magFilter = THREE.LinearFilter;
			textures.push(texture);

			const geometry = new THREE.PlaneGeometry(planeSize, planeSize);
			const material = new THREE.MeshBasicMaterial({
				map: texture,
				transparent: true,
				opacity: 0.85,
				side: THREE.DoubleSide,
				blending: THREE.AdditiveBlending,
				depthWrite: false,
			});

			const mesh = new THREE.Mesh(geometry, material);
			// Stack along Z axis — front to back
			mesh.position.set(0, 0, i * zSpacing - totalDepth / 2);
			// Slight tilt on Y so plates are angled (like the reference image)
			mesh.rotation.y = 0.18;
			scene.add(mesh);
			planeMeshes.push(mesh);
			planeMaterials.push(material);

			// Crimson edge wireframe — matching ConceptScan3D's edge style
			const edgesGeo = new THREE.EdgesGeometry(geometry);
			const edgeMat = new THREE.LineBasicMaterial({
				color: 0xcc0022,
				transparent: true,
				opacity: 0.7,
				blending: THREE.AdditiveBlending,
				depthWrite: false,
			});
			const edgeMesh = new THREE.LineSegments(edgesGeo, edgeMat);
			edgeMesh.position.copy(mesh.position);
			edgeMesh.rotation.copy(mesh.rotation);
			scene.add(edgeMesh);
		}

		// Initial textures
		updateTextures();

		// Build synapses after planes have been added to scene
		rebuildSynapses();

		// Animation loop — breathing opacity only, no rotation, no shimmer flicker
		function animate() {
			animFrame = requestAnimationFrame(animate);
			noiseOffset += 0.02;

			// Breathing opacity pulse — each layer at a different phase
			planeMaterials.forEach((mat, i) => {
				mat.opacity = 0.75 + 0.12 * Math.sin(noiseOffset * 0.8 + i * 0.4);
			});

			renderer.render(scene, camera);
		}
		animate();

		// ResizeObserver — fixes stretched/small canvas
		resizeObserver = new ResizeObserver(() => {
			if (!container || !renderer || !camera) return;
			const cw = container.clientWidth;
			const ch = container.clientHeight;
			if (cw > 0 && ch > 0) {
				renderer.setSize(cw, ch);
				camera.aspect = cw / ch;
				camera.updateProjectionMatrix();
			}
		});
		resizeObserver.observe(container);
	}

	// Reactive: update textures when data changes (force rebuild)
	$: if (matrices.length > 0 && THREE) {
		lastTextureStep = -1; // invalidate cache so rebuild happens
		updateTextures();
	}
	// Reactive: update textures when step changes
	$: if (currentStep >= 0 && THREE) {
		lastTextureStep = -1;
		updateTextures();
	}

	// Reactive: rebuild synapses when battle layers change
	$: if (battleLayers.length > 0 && THREE && planeMeshes.length > 0) {
		rebuildSynapses();
	}

	onMount(() => {
		initScene();
	});

	onDestroy(() => {
		if (animFrame) cancelAnimationFrame(animFrame);
		if (resizeObserver) resizeObserver.disconnect();
		disposeSynapses();
		if (renderer) {
			renderer.dispose();
			renderer.domElement?.remove();
		}
	});
</script>

<div bind:this={container} class="w-full h-full min-h-[300px]"></div>
