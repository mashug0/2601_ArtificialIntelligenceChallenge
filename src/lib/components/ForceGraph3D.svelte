<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import type { ActivationData, GraphNode, GraphData } from '$lib/types';
	import { playbackStore } from '$lib/stores/playback';
	import { selectionStore, layerStore } from '$lib/stores/selection';
	import { buildStaticGraphData, getTokenActivations } from '$lib/utils/graphBuilder';
	import { getNodeColor, getNodeSize, COLORS, getSignalType } from '$lib/utils/colorScales';

	export let data: ActivationData;
	export let sequenceIdx: number = 0;
	/** Optional: dict of node ID → activation weight (0-1) for Battle Arena mode.
	 *  When provided, active nodes follow magma gradient (orange→red→white)
	 *  with size proportional to activation; inactive nodes are invisible dust. */
	export let activeNodes: Record<string, number> | null = null;

	let container: HTMLDivElement;
	let graph: any = null;
	let ForceGraph3DModule: any = null;
	let THREE: any = null;

	// Store mesh references for color updates
	let nodeMeshes: Map<string, any> = new Map();
	let nodeBaseSize: Map<string, number> = new Map();

	// Animation state
	let animationFrame: number;
	let targetColors: Map<string, any> = new Map();
	let targetOpacities: Map<string, number> = new Map();
	let targetScales: Map<string, number> = new Map();
	const LERP_SPEED = 0.12;

	// Build static graph structure (only when data/sequence/layers change)
	$: staticGraphData = buildStaticGraphData(data, sequenceIdx, $layerStore.expanded);

	// Get activations for current token
	$: tokenActivations = getTokenActivations(data, $playbackStore.currentToken, sequenceIdx);

	// When token changes, update target colors (not the graph structure)
	$: if (tokenActivations && THREE) {
		updateTargetColors(tokenActivations);
	}

	// When static graph changes (sequence/layers), rebuild the graph
	$: if (graph && staticGraphData && staticGraphData.nodes.length > 0) {
		rebuildGraph();
	}

	function updateTargetColors(activations: Map<string, { excitatory: number; inhibitory: number }>) {
		activations.forEach((act, nodeId) => {
			const signalType = getSignalType(act.excitatory, act.inhibitory);
			const node = staticGraphData.nodes.find((n) => n.id === nodeId);
			const isSpecialist = node?.is_specialist || false;

			const colorStr = getNodeColor(signalType, act.excitatory, act.inhibitory, isSpecialist);
			const targetColor = new THREE.Color(colorStr);
			targetColors.set(nodeId, targetColor);

			const isActive = act.excitatory > 0.05 || act.inhibitory > 0.05;
			targetOpacities.set(nodeId, isActive ? 0.9 : 0.4);

			const targetSize = getNodeSize(act.excitatory, act.inhibitory);
			const baseSize = nodeBaseSize.get(nodeId) || 5;
			targetScales.set(nodeId, targetSize / baseSize);
		});
	}

	/**
	 * Magma gradient: maps activation weight (0-1) to orange→red→white
	 *   < 0.1 : Invisible / Dark Grey
	 *   0.1-0.5: Dark Orange
	 *   0.5-0.8: Bright Orange
	 *   > 0.8 : Glowing Red/White
	 */
	function magmaColor(val: number): any {
		if (!THREE) return null;
		if (val < 0.1) return new THREE.Color(0x1a1a1a);        // Invisible grey
		if (val < 0.5) {
			// Dark Orange: lerp from 0x8B4000 → 0xFF6600
			const t = (val - 0.1) / 0.4;
			const c = new THREE.Color(0x8B4000);
			c.lerp(new THREE.Color(0xFF6600), t);
			return c;
		}
		if (val < 0.8) {
			// Bright Orange: lerp from 0xFF6600 → 0xFF2200
			const t = (val - 0.5) / 0.3;
			const c = new THREE.Color(0xFF6600);
			c.lerp(new THREE.Color(0xFF2200), t);
			return c;
		}
		// Glowing Red/White: lerp from 0xFF2200 → 0xFFCCAA
		const t = (val - 0.8) / 0.2;
		const c = new THREE.Color(0xFF2200);
		c.lerp(new THREE.Color(0xFFCCAA), t);
		return c;
	}

	function applyActiveNodesOverlay(weights: Record<string, number>) {
		if (!THREE) return;
		staticGraphData.nodes.forEach((node) => {
			const val = weights[node.id];
			if (val !== undefined && val > 0) {
				// Active: magma color gradient, size explodes with activation
				targetColors.set(node.id, magmaColor(val));
				targetOpacities.set(node.id, val < 0.1 ? 0.0 : 0.5 + val * 0.5);
				// Inactive=0.5 (dust), Active = 2 + val*5 → big pulse for high activation
				const baseSize = nodeBaseSize.get(node.id) || 5;
				targetScales.set(node.id, (2 + val * 5) / baseSize);
			} else {
				// Ghost: tiny invisible dust
				targetColors.set(node.id, new THREE.Color(0x111111));
				targetOpacities.set(node.id, 0.0);
				const baseSize = nodeBaseSize.get(node.id) || 5;
				targetScales.set(node.id, 0.5 / baseSize);
			}
		});
	}

	function applyGhostDefaults() {
		// Ghost logic: default ALL nodes to invisible until activeNodes illuminates them
		if (!THREE) return;
		staticGraphData.nodes.forEach((node) => {
			targetColors.set(node.id, new THREE.Color(0x111111));
			targetOpacities.set(node.id, 0.0);
			targetScales.set(node.id, 0.3);
		});
	}

	// When activeNodes prop is set, override normal color logic
	$: if (activeNodes !== null && THREE) {
		if (Object.keys(activeNodes).length > 0) {
			applyActiveNodesOverlay(activeNodes);
		} else {
			applyGhostDefaults();
		}
	}

	function animateColors() {
		nodeMeshes.forEach((mesh, nodeId) => {
			const targetColor = targetColors.get(nodeId);
			const targetOpacity = targetOpacities.get(nodeId);
			const targetScale = targetScales.get(nodeId);

			if (targetColor && mesh.material) {
				mesh.material.color.lerp(targetColor, LERP_SPEED);
			}

			if (targetOpacity !== undefined && mesh.material) {
				mesh.material.opacity += (targetOpacity - mesh.material.opacity) * LERP_SPEED;
			}

			if (targetScale !== undefined) {
				const currentScale = mesh.scale.x;
				const newScale = currentScale + (targetScale - currentScale) * LERP_SPEED;
				mesh.scale.setScalar(newScale);
			}

			// Update emissive for specialists
			const act = tokenActivations.get(nodeId);
			const node = staticGraphData.nodes.find((n) => n.id === nodeId);
			if (node?.is_specialist && mesh.material && act) {
				const targetEmissive = act.excitatory > 0.1 ? 0.4 : 0;
				mesh.material.emissiveIntensity +=
					(targetEmissive - mesh.material.emissiveIntensity) * LERP_SPEED;
			}
		});

		animationFrame = requestAnimationFrame(animateColors);
	}

	function rebuildGraph() {
		if (!graph) return;

		// Clear mesh references
		nodeMeshes.clear();
		nodeBaseSize.clear();

		// Set new graph data
		graph.graphData(staticGraphData);

		// Re-apply initial colors based on current token
		setTimeout(() => {
			updateTargetColors(tokenActivations);
		}, 100);
	}

	async function initGraph() {
		if (!browser || !container) return;

		const module = await import('3d-force-graph');
		ForceGraph3DModule = module.default;
		THREE = await import('three');

		graph = ForceGraph3DModule()(container)
			.graphData(staticGraphData)
			.nodeThreeObject((node: GraphNode) => {
				const act = tokenActivations.get(node.id) || { excitatory: 0, inhibitory: 0 };
				const signalType = getSignalType(act.excitatory, act.inhibitory);

				const size = getNodeSize(act.excitatory, act.inhibitory);
				const baseSize = Math.max(size, 5);
				nodeBaseSize.set(node.id, baseSize);

				const color = getNodeColor(signalType, act.excitatory, act.inhibitory, node.is_specialist);

				// Slightly higher segment count + shinier material for deep-glow look
				const geometry = new THREE.SphereGeometry(baseSize, 20, 20);
				const material = new THREE.MeshPhongMaterial({
					color,
					emissive: node.is_specialist ? COLORS.specialist : 0x000000,
					emissiveIntensity: node.is_specialist && act.excitatory > 0.1 ? 0.6 : 0.15,
					shininess: 80,
					transparent: true,
					opacity: signalType === 'inactive' ? 0.35 : 0.95
				});

				const mesh = new THREE.Mesh(geometry, material);

				// Add glow ring for specialists
				if (node.is_specialist) {
					const ringGeometry = new THREE.RingGeometry(baseSize * 1.3, baseSize * 1.5, 32);
					const ringMaterial = new THREE.MeshBasicMaterial({
						color: COLORS.specialist,
						transparent: true,
						opacity: act.excitatory > 0.1 ? 0.5 : 0.1,
						side: THREE.DoubleSide
					});
					const ring = new THREE.Mesh(ringGeometry, ringMaterial);
					mesh.add(ring);
				}

				// Store mesh reference for color updates
				nodeMeshes.set(node.id, mesh);

				return mesh;
			})
			.nodeThreeObjectExtend(false)
			.linkWidth((link: any) => {
				// Battle mode: links between active nodes are thick lightning bolts
				if (activeNodes !== null) {
					const srcId = typeof link.source === 'object' ? link.source.id : link.source;
					const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
					const srcVal = activeNodes[srcId] ?? 0;
					const tgtVal = activeNodes[tgtId] ?? 0;
					if (srcVal > 0 && tgtVal > 0) {
						return 1 + ((srcVal + tgtVal) / 2) * 3;
					}
					return 0; // Invisible if either node is inactive
				}
				return Math.max(0.4, link.weight * 2.5);
			})
			// Slightly brighter links for dark scene
			.linkOpacity(0.45)
			.linkColor((link: any) => {
				// Battle mode: orange lightning between active nodes
				if (activeNodes !== null) {
					const srcId = typeof link.source === 'object' ? link.source.id : link.source;
					const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
					const srcVal = activeNodes[srcId] ?? 0;
					const tgtVal = activeNodes[tgtId] ?? 0;
					if (srcVal > 0 && tgtVal > 0) {
						const avg = (srcVal + tgtVal) / 2;
						const r = Math.round(255);
						const g = Math.round(80 + 80 * avg);
						const b = Math.round(10);
						return `rgba(${r},${g},${b},${0.3 + avg * 0.7})`;
					}
					return 'rgba(0,0,0,0)'; // Invisible
				}

				// Default: subtle cyan→magenta blend based on weight
				const w = link.weight ?? 0.3;
				const alpha = 0.25 + w * 0.4;
				const r = Math.round(56 + 80 * w);
				const g = Math.round(189 - 40 * w);
				const b = Math.round(248 - 80 * w);
				return `rgba(${r},${g},${b},${alpha})`;
			})
			.linkDirectionalParticles(1)
			.linkDirectionalParticleWidth(1)
			.linkDirectionalParticleSpeed(0.003)
			// Match deep-glow dashboard base background
			.backgroundColor(COLORS.background)
			.d3Force('charge', null)
			.d3Force('center', null)
			.d3VelocityDecay(0.4)
			.onNodeHover((node: GraphNode | null) => {
				selectionStore.setHovered(node);
				container.style.cursor = node ? 'pointer' : 'default';
			})
			.onNodeClick((node: GraphNode) => {
				selectionStore.toggleSelected(node);
			});

		// Custom forces for layer positioning
		setupCustomForces();

		// Setup lighting tuned for dark deep-glow scene
		const scene = graph.scene();
		const ambientLight = new THREE.AmbientLight(0x0f172a, 0.9);
		scene.add(ambientLight);

		const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
		directionalLight.position.set(100, 100, 100);
		scene.add(directionalLight);

		// Camera position
		graph.cameraPosition({ x: 0, y: 0, z: 600 }, { x: 0, y: 0, z: 0 }, 0);

		// Initialize target colors — ghost mode if activeNodes prop present
		if (activeNodes !== null) {
			// Battle Arena mode: void black background (same deep-glow tone)
			graph.backgroundColor(COLORS.background);
			applyGhostDefaults();
		} else {
			updateTargetColors(tokenActivations);
		}

		// Start animation loop
		animateColors();

		// ResizeObserver — fixes stretched/small canvas in split pane
		const resizeObs = new ResizeObserver(() => {
			if (!container || !graph) return;
			const cw = container.clientWidth;
			const ch = container.clientHeight;
			if (cw > 0 && ch > 0) {
				graph.width(cw);
				graph.height(ch);
			}
		});
		resizeObs.observe(container);
	}

	function setupCustomForces() {
		if (!graph) return;

		// Force to position nodes by layer along Z-axis
		const layerForce = (alpha: number) => {
			const LAYER_SPACING = 80;
			const nodes = graph.graphData().nodes as GraphNode[];

			nodes.forEach((node: GraphNode) => {
				const targetZ =
					node.layer * LAYER_SPACING - ((data.metadata.n_layers - 1) * LAYER_SPACING) / 2;
				node.vz = (node.vz || 0) + (targetZ - (node.z || 0)) * alpha * 0.1;
			});
		};

		// Repulsion between nodes in same layer
		const intraLayerRepulsion = (alpha: number) => {
			const nodes = graph.graphData().nodes as GraphNode[];
			const strength = -50;

			for (let i = 0; i < nodes.length; i++) {
				for (let j = i + 1; j < nodes.length; j++) {
					if (nodes[i].layer === nodes[j].layer) {
						const dx = (nodes[j].x || 0) - (nodes[i].x || 0);
						const dy = (nodes[j].y || 0) - (nodes[i].y || 0);
						const dist = Math.sqrt(dx * dx + dy * dy) || 1;
						const force = (strength * alpha) / (dist * dist);

						nodes[i].vx = (nodes[i].vx || 0) - (dx / dist) * force;
						nodes[i].vy = (nodes[i].vy || 0) - (dy / dist) * force;
						nodes[j].vx = (nodes[j].vx || 0) + (dx / dist) * force;
						nodes[j].vy = (nodes[j].vy || 0) + (dy / dist) * force;
					}
				}
			}
		};

		graph.d3Force('layer', layerForce);
		graph.d3Force('intraLayer', intraLayerRepulsion);
	}

	// Fit graph to frame with smooth animation
	function fitToFrame(duration: number = 1000) {
		if (!graph) return;
		graph.zoomToFit(duration, 50); // 50px padding
	}

	// Reset camera to initial position
	function resetView() {
		if (!graph) return;
		graph.cameraPosition({ x: 0, y: 0, z: 600 }, { x: 0, y: 0, z: 0 }, 1000);
	}

	onMount(() => {
		initGraph();
	});

	onDestroy(() => {
		if (animationFrame) {
			cancelAnimationFrame(animationFrame);
		}
		if (graph) {
			graph._destructor?.();
		}
	});
</script>

<div class="graph-wrapper relative w-full h-full min-h-[400px]">
	<div bind:this={container} class="graph-container w-full h-full"></div>

	<!-- Camera Controls -->
	<div class="absolute top-4 right-4 flex flex-col gap-2 z-10">
		<button
			on:click={() => fitToFrame()}
			class="bg-surface/80 hover:bg-surface-hover/90 px-3 py-2 rounded-md border border-border/60 shadow-sm text-xs font-medium text-foreground flex items-center gap-2 transition-colors"
			title="Fit graph to frame"
		>
			<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
				/>
			</svg>
			Fit
		</button>
		<button
			on:click={() => resetView()}
			class="bg-surface/80 hover:bg-surface-hover/90 px-3 py-2 rounded-md border border-border/60 shadow-sm text-xs font-medium text-foreground flex items-center gap-2 transition-colors"
			title="Reset camera view"
		>
			<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"
				/>
			</svg>
			Reset
		</button>
	</div>
</div>

<style>
	.graph-wrapper {
		border-radius: 4px;
		overflow: hidden;
	}

	.graph-container {
		/* Deep-glow background to match dashboard */
		background:
			radial-gradient(circle at top, hsla(186, 100%, 50%, 0.08), transparent),
			radial-gradient(circle at bottom, hsla(324, 100%, 53%, 0.08), transparent),
			linear-gradient(180deg, #020617 0%, #020314 100%);
		width: 100%;
		height: 100%;
	}

	.graph-container :global(canvas) {
		display: block;
	}
</style>
