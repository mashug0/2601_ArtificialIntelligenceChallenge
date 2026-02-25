<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';

	/** Layer data from concept trace */
	export let layers: {
		layer: number;
		active_neurons: number;
		total_neurons: number;
		sparsity: number;
		top_neuron_ids: number[];
		top_neuron_values: number[];
		mean_activation: number;
	}[] = [];

	/** Accent color for current concept group */
	export let accentColor: string = '#00ffff';

	/** Callback when a layer is hovered (popped out) — sends layer index or -1 */
	export let onLayerHover: (layerIdx: number) => void = () => {};

	let container: HTMLDivElement;
	let THREE: any;
	let scene: any, camera: any, renderer: any;
	let raycaster: any, mouse: any;
	let layerMeshes: any[] = [];
	let textures: any[] = [];
	let edgeLines: any[] = [];
	let laserMesh: any;
	let laserY = -3;
	let laserDirection = 1;
	let animFrame: number;
	let hoveredLayerIdx = -1;
	let popOutTargets: Map<number, number> = new Map(); // layer idx -> target X offset
	let currentPopOuts: Map<number, number> = new Map(); // layer idx -> current X offset
	let ro: ResizeObserver;
	let disposed = false;

	// Grid dimensions for mapping neuron IDs to 2D
	const GRID_W = 32;
	const GRID_H = 16; // 32*16 = 512 neurons max
	const TEX_SIZE = 256;

	// Layer geometry
	const PLANE_WIDTH = 4;
	const PLANE_HEIGHT = 2.5;
	const LAYER_GAP = 1.8;

	function createNeuronTexture(layerData: typeof layers[0]): any {
		const canvas = document.createElement('canvas');
		canvas.width = TEX_SIZE;
		canvas.height = TEX_SIZE;
		const ctx = canvas.getContext('2d')!;

		// Fill black (transparent base)
		ctx.fillStyle = '#000000';
		ctx.fillRect(0, 0, TEX_SIZE, TEX_SIZE);

		const cellW = TEX_SIZE / GRID_W;
		const cellH = TEX_SIZE / GRID_H;

		// Normalize values
		const maxVal = Math.max(...layerData.top_neuron_values, 0.001);

		// Build node positions for connecting lines
		const positions: { x: number; y: number; val: number }[] = [];
		layerData.top_neuron_ids.forEach((nId, i) => {
			const val = layerData.top_neuron_values[i] / maxVal;
			const gx = nId % GRID_W;
			const gy = Math.floor(nId / GRID_W) % GRID_H;
			positions.push({
				x: gx * cellW + cellW / 2,
				y: gy * cellH + cellH / 2,
				val
			});
		});

		// Draw luminous lines between nearby nodes (within ~3 cells)
		const linkDist = Math.sqrt((cellW * 3) ** 2 + (cellH * 3) ** 2);
		for (let i = 0; i < positions.length; i++) {
			for (let j = i + 1; j < positions.length; j++) {
				const dx = positions[j].x - positions[i].x;
				const dy = positions[j].y - positions[i].y;
				const d = Math.sqrt(dx * dx + dy * dy);
				if (d < linkDist) {
					const alpha = 0.15 + 0.2 * (1 - d / linkDist) * (positions[i].val + positions[j].val) / 2;
					ctx.strokeStyle = `rgba(255, 200, 100, ${alpha})`;
					ctx.lineWidth = 1.5;
					ctx.globalCompositeOperation = 'lighter';
					ctx.beginPath();
					ctx.moveTo(positions[i].x, positions[i].y);
					ctx.lineTo(positions[j].x, positions[j].y);
					ctx.stroke();
				}
			}
		}
		ctx.globalCompositeOperation = 'source-over';

		// Draw glowing orbs for each active neuron
		positions.forEach(({ x: cx, y: cy, val }) => {
			// Outer halo — wide diffuse glow
			const haloR = cellW * (2.5 + val * 3);
			const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, haloR);
			halo.addColorStop(0, `rgba(255, 220, 150, ${0.3 + val * 0.4})`);
			halo.addColorStop(0.5, `rgba(255, 180, 80, ${0.1 + val * 0.2})`);
			halo.addColorStop(1, 'rgba(0,0,0,0)');
			ctx.fillStyle = halo;
			ctx.fillRect(cx - haloR, cy - haloR, haloR * 2, haloR * 2);

			// Inner core — bright orb
			const coreR = cellW * (0.8 + val * 1.2);
			const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
			grad.addColorStop(0, `rgba(255, 255, 255, ${0.9 + val * 0.1})`);
			grad.addColorStop(0.3, `rgba(255, 230, 150, ${0.8 + val * 0.2})`);
			grad.addColorStop(0.6, `rgba(255, 180, 80, ${0.4 + val * 0.4})`);
			grad.addColorStop(1, 'rgba(0,0,0,0)');
			ctx.fillStyle = grad;
			ctx.fillRect(cx - coreR, cy - coreR, coreR * 2, coreR * 2);
		});

		const tex = new THREE.CanvasTexture(canvas);
		tex.needsUpdate = true;
		return tex;
	}

	function buildScene() {
		if (!THREE || !container || layers.length === 0) return;

		scene = new THREE.Scene();
		scene.background = new THREE.Color(0x03060c);

		// Deep space fog for depth
		scene.fog = new THREE.FogExp2(0x020408, 0.04);

		const w = container.clientWidth || 600;
		const h = container.clientHeight || 500;
		camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
		camera.position.set(0, 0, 9);
		camera.lookAt(0, 0, 0);

		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setSize(w, h);
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.toneMapping = THREE.ACESFilmicToneMapping;
		renderer.toneMappingExposure = 1.2;
		container.appendChild(renderer.domElement);

		// Raycaster for hover
		raycaster = new THREE.Raycaster();
		mouse = new THREE.Vector2(-999, -999);

		// Lighting — medical/clinical cold teal
		const ambient = new THREE.AmbientLight(0x1a3a4a, 0.6);
		scene.add(ambient);
		const dirLight = new THREE.DirectionalLight(0x88ccdd, 0.8);
		dirLight.position.set(3, 5, 5);
		scene.add(dirLight);
		const rimLight = new THREE.DirectionalLight(0x00ffff, 0.2);
		rimLight.position.set(-3, -2, 4);
		scene.add(rimLight);

		// Calculate vertical centering
		const totalHeight = (layers.length - 1) * LAYER_GAP;
		const startY = totalHeight / 2;

		layerMeshes = [];
		textures = [];
		edgeLines = [];
		currentPopOuts = new Map();
		popOutTargets = new Map();

		layers.forEach((layerData, i) => {
			const y = startY - i * LAYER_GAP;

			// Glass slide geometry
			const geo = new THREE.PlaneGeometry(PLANE_WIDTH, PLANE_HEIGHT);

			// Neuron texture
			const tex = createNeuronTexture(layerData);
			textures.push(tex);

			// Glassmorphic plane — transmission + cyan edge glow
			const mat = new THREE.MeshPhysicalMaterial({
				transmission: 0.92,
				roughness: 0.05,
				thickness: 0.08,
				color: new THREE.Color(0xe0f7fa),
				emissive: new THREE.Color(0x00ffff),
				emissiveIntensity: 0.08,
				side: THREE.DoubleSide,
				transparent: true,
				opacity: 0.75,
				depthWrite: false,
			});

			const mesh = new THREE.Mesh(geo, mat);
			mesh.position.set(0, y, 0);
			mesh.rotation.x = -0.3; // Slight tilt for 3D depth
			mesh.userData = { layerIdx: i };
			scene.add(mesh);

			// Neuron overlay plane (additive blending)
			const overlayGeo = new THREE.PlaneGeometry(PLANE_WIDTH, PLANE_HEIGHT);
			const overlayMat = new THREE.MeshBasicMaterial({
				map: tex,
				transparent: true,
				blending: THREE.AdditiveBlending,
				depthWrite: false,
				opacity: 0.6,
				side: THREE.DoubleSide,
			});
			const overlayMesh = new THREE.Mesh(overlayGeo, overlayMat);
			overlayMesh.position.copy(mesh.position);
			overlayMesh.rotation.copy(mesh.rotation);
			overlayMesh.position.z += 0.01;
			scene.add(overlayMesh);

			// Luminous cyan edge wireframe — glassmorphic border glow
			const edgesGeo = new THREE.EdgesGeometry(geo);
			const edgeMat = new THREE.LineBasicMaterial({
				color: 0x00ffff,
				transparent: true,
				opacity: 0.6,
			});
			const edgeLine = new THREE.LineSegments(edgesGeo, edgeMat);
			edgeLine.position.copy(mesh.position);
			edgeLine.rotation.copy(mesh.rotation);
			scene.add(edgeLine);
			edgeLines.push(edgeLine);

			layerMeshes.push({ glass: mesh, overlay: overlayMesh, edge: edgeLine, baseY: y });
			currentPopOuts.set(i, 0);
			popOutTargets.set(i, 0);
		});

		// Scanner — thick glowing cyan beam
		const laserGeo = new THREE.PlaneGeometry(PLANE_WIDTH + 2, 0.22);
		const laserMat = new THREE.MeshBasicMaterial({
			color: 0x00ffff,
			transparent: true,
			opacity: 0.95,
			blending: THREE.AdditiveBlending,
			side: THREE.DoubleSide,
		});
		laserMesh = new THREE.Mesh(laserGeo, laserMat);
		laserMesh.rotation.x = -0.3;
		laserMesh.position.set(0, startY + 1, 0.1);
		laserY = startY + 1;
		scene.add(laserMesh);

		// Event listeners
		renderer.domElement.addEventListener('mousemove', onMouseMove);
		renderer.domElement.addEventListener('mouseleave', onMouseLeave);

		// ResizeObserver
		ro = new ResizeObserver(() => {
			if (!renderer || !camera || !container || disposed) return;
			const rw = container.clientWidth;
			const rh = container.clientHeight;
			camera.aspect = rw / rh;
			camera.updateProjectionMatrix();
			renderer.setSize(rw, rh);
		});
		ro.observe(container);

		animate();
	}

	function onMouseMove(event: MouseEvent) {
		if (!renderer || !container) return;
		const rect = renderer.domElement.getBoundingClientRect();
		mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
	}

	function onMouseLeave() {
		mouse.x = -999;
		mouse.y = -999;
		hoveredLayerIdx = -1;
		onLayerHover(-1);
		// Reset all pop-out targets
		for (let i = 0; i < layers.length; i++) {
			popOutTargets.set(i, 0);
		}
	}

	function animate() {
		if (disposed) return;
		animFrame = requestAnimationFrame(animate);
		if (!scene || !camera || !renderer) return;

		const totalHeight = (layers.length - 1) * LAYER_GAP;
		const startY = totalHeight / 2;
		const endY = -totalHeight / 2;

		// Move laser up/down
		laserY += laserDirection * 0.015;
		if (laserY > startY + 1.5) laserDirection = -1;
		if (laserY < endY - 1.5) laserDirection = 1;
		if (laserMesh) laserMesh.position.y = laserY;

		// Raycaster for hover detection
		raycaster.setFromCamera(mouse, camera);
		const glassMeshes = layerMeshes.map((l: any) => l.glass);
		const intersects = raycaster.intersectObjects(glassMeshes);

		let newHovered = -1;
		if (intersects.length > 0) {
			newHovered = intersects[0].object.userData.layerIdx;
		}

		if (newHovered !== hoveredLayerIdx) {
			hoveredLayerIdx = newHovered;
			onLayerHover(hoveredLayerIdx);

			// Set pop-out targets
			for (let i = 0; i < layers.length; i++) {
				if (i === hoveredLayerIdx) {
					popOutTargets.set(i, -2.5); // Slide out left like a drawer
				} else {
					popOutTargets.set(i, 0);
				}
			}
		}

		// Update each layer
		layerMeshes.forEach((lm: any, i: number) => {
			// Smooth pop-out interpolation
			const target = popOutTargets.get(i) ?? 0;
			const current = currentPopOuts.get(i) ?? 0;
			const next = current + (target - current) * 0.08;
			currentPopOuts.set(i, next);

			const xOff = next;
			lm.glass.position.x = xOff;
			lm.overlay.position.x = xOff;
			lm.edge.position.x = xOff;
			lm.overlay.position.z = 0.01 + xOff * 0; // keep overlay aligned

			// MRI scan proximity effect
			const dist = Math.abs(laserY - lm.baseY);
			const proximity = Math.max(0, 1 - dist / 1.2);

			// Hovered layer gets full brightness, others dim
			const isHovered = i === hoveredLayerIdx;
			const baseOpacity = hoveredLayerIdx >= 0 ? (isHovered ? 0.9 : 0.1) : 0.7;

			// Glass opacity
			lm.glass.material.opacity = baseOpacity;

			// Overlay (neuron texture) — boost when laser passes or hovered
			const overlayBoost = isHovered ? 1.0 : proximity;
			lm.overlay.material.opacity = 0.3 + overlayBoost * 0.7;

			// Scale neurons slightly when laser passes
			const sc = 1 + proximity * 0.08;
			lm.overlay.scale.set(sc, sc, 1);

			// Edge color: cyan base, bright cyan when laser passes or hovered
			if (isHovered) {
				lm.edge.material.color.setHex(0x00ffff);
				lm.edge.material.opacity = 1.0;
			} else if (proximity > 0.3) {
				lm.edge.material.color.setHex(0x00ffff);
				lm.edge.material.opacity = 0.6 + proximity * 0.4;
			} else {
				lm.edge.material.color.setHex(0x00cccc);
				lm.edge.material.opacity = hoveredLayerIdx >= 0 ? 0.15 : 0.5;
			}
		});

		renderer.render(scene, camera);
	}

	onMount(async () => {
		if (!browser) return;
		const t = await import('three');
		THREE = t;
		buildScene();
	});

	onDestroy(() => {
		disposed = true;
		if (animFrame) cancelAnimationFrame(animFrame);
		if (renderer) {
			renderer.domElement.removeEventListener('mousemove', onMouseMove);
			renderer.domElement.removeEventListener('mouseleave', onMouseLeave);
			renderer.dispose();
		}
		if (ro) ro.disconnect();
		textures.forEach((tex: any) => tex.dispose());
	});

	// Rebuild when layers change
	$: if (THREE && layers.length > 0 && container) {
		// Clean up old scene
		if (renderer && container.contains(renderer.domElement)) {
			renderer.domElement.removeEventListener('mousemove', onMouseMove);
			renderer.domElement.removeEventListener('mouseleave', onMouseLeave);
			container.removeChild(renderer.domElement);
		}
		textures.forEach((tex: any) => tex?.dispose());
		buildScene();
	}
</script>

<div bind:this={container} class="w-full h-full"></div>
