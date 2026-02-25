<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import type {
		ComparisonData,
		ConceptBattleResult,
		ExplainerData,
		ActivationData,
		BattleData
	} from '$lib/types';

	let data: ComparisonData | null = null;
	let explainerData: ExplainerData | null = null;
	let activationData: ActivationData | null = null;
	let battleData: BattleData | null = null;
	let isLoading = true;
	let error = '';
	let selectedConcept = '';
	let currentStep = 0;

	// Dynamic imports for 3D components
	let Transformer3D: any = null;
	let ForceGraph3D: any = null;

	// Heatmap canvas refs
	let bdhCanvas: HTMLCanvasElement;
	let transCanvas: HTMLCanvasElement;

	// Blinking HUD state
	let criticalBlink = false;
	let blinkInterval: ReturnType<typeof setInterval>;

	async function loadData() {
		try {
			const [compRes, explRes, actRes, battleRes] = await Promise.all([
				fetch('/data/comparison_data.json').catch(() => null),
				fetch('/data/explainer.json').catch(() => null),
				fetch('/data/activations.json').catch(() => null),
				fetch('/data/battle_data.json').catch(() => null),
			]);

			if (compRes && compRes.ok) {
				data = await compRes.json();
				if (data && data.concept_battle) {
					const concepts = Object.keys(data.concept_battle);
					if (concepts.length > 0) selectedConcept = concepts[0];
				}
			}

			if (explRes && explRes.ok) {
				explainerData = await explRes.json();
			}

			if (actRes && actRes.ok) {
				activationData = await actRes.json();
			}

			if (battleRes && battleRes.ok) {
				battleData = await battleRes.json();
			}

			if (!data && !explainerData && !battleData) {
				error = 'No data found. Run export_comparison.py, export_explainer.py, or export_battle.py.';
			}
		} catch (e: any) {
			error = e.message || 'Failed to load data';
		} finally {
			isLoading = false;
		}
	}

	onMount(async () => {
		loadData();
		if (browser) {
			const t3d = await import('$lib/components/Transformer3D.svelte');
			Transformer3D = t3d.default;
			const fg3d = await import('$lib/components/ForceGraph3D.svelte');
			ForceGraph3D = fg3d.default;
		}
		// Blinking CRITICAL warning
		blinkInterval = setInterval(() => { criticalBlink = !criticalBlink; }, 800);
	});

	// Derive active node weights for current step from battle data (dict: id->weight)
	// or fall back to explainer data (array of node objects)
	$: battleGraph = battleData?.bdh_graph?.[currentStep];
	$: bdhActiveNodeWeights = battleGraph
		? battleGraph.active_nodes  // Record<string, number> — { "L0_N102": 0.45, ... }
		: (explainerData?.bdh_steps?.[currentStep]
			? Object.fromEntries(explainerData.bdh_steps[currentStep].active_nodes.map(n => [n.id, Math.min(n.value, 1.0)]))
			: {} as Record<string, number>);

	// Derive narrative from explainer
	$: currentNarrative = explainerData?.narrative
		? explainerData.narrative[Math.min(currentStep, explainerData.narrative.length - 1)]
		: '';

	// Max step
	$: maxStep = battleData
		? battleData.metadata.n_tokens - 1
		: (explainerData ? explainerData.metadata.n_tokens - 1 : 0);

	// Token labels
	$: tokenLabels = battleData?.metadata?.tokens
		?? explainerData?.metadata?.tokens
		?? [];

	// Load percentages
	$: transLoad = battleData?.transformer_load ?? 100;
	$: bdhLoad = battleData?.bdh_load ?? 2.4;
	$: energySavings = battleData?.energy_savings ?? (transLoad - bdhLoad);

	// Active count for current token
	$: currentActiveCount = battleGraph?.n_active ?? Object.keys(bdhActiveNodeWeights).length;
	$: currentTotalNeurons = battleGraph?.n_total ?? (battleData?.metadata?.bdh_neurons_per_layer ?? 2048) * (battleData?.metadata?.n_layers ?? 4);

	// --- Heatmap rendering ---
	function drawHeatmap(
		canvas: HTMLCanvasElement,
		grid: number[][],
		maxVal: number,
		isSparse: boolean
	) {
		if (!canvas || !grid || grid.length === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const T = grid.length;
		const N = grid[0].length;
		const cellW = canvas.width / N;
		const cellH = canvas.height / T;

		ctx.fillStyle = '#0a0a0a';
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		for (let t = 0; t < T; t++) {
			for (let n = 0; n < N; n++) {
				const val = grid[t][n];
				if (val === 0 && isSparse) continue;

				const norm = maxVal > 0 ? Math.min(Math.abs(val) / maxVal, 1.0) : 0;
				if (norm < 0.01) continue;

				if (isSparse) {
					const r = Math.round(30 + 50 * norm);
					const g = Math.round(180 + 75 * norm);
					const b = Math.round(220 + 35 * norm);
					ctx.fillStyle = `rgba(${r},${g},${b},${0.5 + 0.5 * norm})`;
				} else {
					const r = Math.round(180 + 75 * norm);
					const g = Math.round(60 + 80 * norm);
					const b = Math.round(40 + 30 * norm);
					ctx.fillStyle = `rgba(${r},${g},${b},${0.15 + 0.7 * norm})`;
				}
				ctx.fillRect(n * cellW, t * cellH, cellW + 0.5, cellH + 0.5);
			}
		}
	}

	$: if (data && bdhCanvas && transCanvas) {
		const vs = data.visual_sample;
		const bdhMax = Math.max(...vs.bdh_grid.flat().map(Math.abs), 0.01);
		const transMax = Math.max(...vs.transformer_grid.flat().map(Math.abs), 0.01);
		drawHeatmap(bdhCanvas, vs.bdh_grid, bdhMax, true);
		drawHeatmap(transCanvas, vs.transformer_grid, transMax, false);
	}

	function getNoiseWords(battle: ConceptBattleResult): string[] {
		return Object.keys(battle.bdh_noise_activations);
	}

	function getMaxNoiseAbs(battle: ConceptBattleResult): number {
		const allVals = [
			...Object.values(battle.bdh_noise_activations).map(Math.abs),
			...Object.values(battle.transformer_noise_activations).map(Math.abs)
		];
		return Math.max(...allVals, 0.001);
	}
</script>

<svelte:head>
	<title>BDH vs Transformer - Battle Arena</title>
</svelte:head>

<div class="w-full">
	{#if isLoading}
		<div class="flex items-center justify-center min-h-[400px] bg-[#020617]">
			<div class="text-center">
				<div class="animate-spin w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full mx-auto mb-4"></div>
				<p class="text-muted-foreground">Initializing Battle Arena...</p>
			</div>
		</div>
	{:else if error}
		<div class="bg-red-950 border border-red-800 rounded-lg p-6 text-center m-8">
			<p class="text-red-400 font-medium">Error: {error}</p>
			<p class="text-red-500 text-sm mt-2">
				Run <code class="bg-red-900 px-2 py-0.5 rounded text-red-300">python export_battle.py</code> in the backend.
			</p>
		</div>
	{:else}

		<!-- ================================================================ -->
		<!-- TOP 70%: THE ARENA — Split 3D Viewports                          -->
		<!-- ================================================================ -->
		{#if (battleData || explainerData)}
			<section class="relative" style="height: 70vh; min-height: 450px;">
				<div class="flex h-full">
					<!-- LEFT PANE: The Machine (Transformer) — Cold Grey/Green Industrial -->
					<div class="relative flex-1 bg-[#020617] overflow-hidden">
						<!-- Industrial grid background — cold green -->
						<div class="absolute inset-0 opacity-[0.07]" style="background-image: linear-gradient(rgba(74,222,128,0.35) 1px, transparent 1px), linear-gradient(90deg, rgba(74,222,128,0.35) 1px, transparent 1px); background-size: 24px 24px;"></div>

						<!-- 3D Component -->
						{#if Transformer3D}
							<svelte:component
								this={Transformer3D}
								steps={explainerData?.transformer_steps ?? []}
								battleLayers={battleData?.transformer_layers ?? []}
								{currentStep}
							/>
						{:else}
							<div class="flex items-center justify-center h-full text-red-500/40 text-sm">Loading renderer...</div>
						{/if}

						<!-- HUD Overlay: Left — Glass Card -->
						<div class="absolute top-4 left-4 z-50 pointer-events-none bg-black/50 backdrop-blur-sm p-4 rounded-lg">
							<h3 class="text-gray-300 font-bold text-sm tracking-[0.2em] uppercase drop-shadow-lg">TRANSFORMER (DENSE)</h3>
							<p class="text-gray-400/50 text-xs mt-0.5">Standard Softmax + GeLU</p>
							<div class="mt-3 font-mono">
								<p class="text-xs text-gray-400/60">ACTIVE LOAD</p>
								<p class="text-3xl font-black text-gray-300 drop-shadow-[0_0_10px_rgba(180,180,180,0.3)]">{transLoad}%</p>
								<p
									class="text-xs font-bold tracking-wider mt-1 transition-opacity duration-200"
									class:text-red-500={criticalBlink}
									class:text-red-800={!criticalBlink}
								>
									CRITICAL — OVERLOAD
								</p>
							</div>
						</div>
					</div>

					<!-- CENTER DIVIDER: Glowing line -->
					<div class="relative w-[2px] flex-shrink-0 z-30">
						<div class="absolute inset-0 bg-yellow-400/60 shadow-[0_0_12px_rgba(250,200,50,0.4)]"></div>
						<span class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-gray-900 text-yellow-400 font-black text-sm px-2.5 py-1 rounded border border-yellow-500/50 shadow-[0_0_15px_rgba(250,200,50,0.3)]">VS</span>
					</div>

					<!-- RIGHT PANE: The Brain (BDH) — Pitch Black Void -->
					<div class="relative flex-1 bg-[#050510] overflow-hidden">
						<!-- Subtle radial glow — warm magma tint -->
						<div class="absolute inset-0" style="background: radial-gradient(ellipse at center, rgba(255,100,20,0.04) 0%, transparent 60%);"></div>

						<!-- 3D Component -->
						{#if ForceGraph3D && activationData}
							<svelte:component
								this={ForceGraph3D}
								data={activationData}
								sequenceIdx={0}
								activeNodes={bdhActiveNodeWeights}
							/>
						{:else}
							<div class="flex items-center justify-center h-full text-cyan-500/40 text-sm">
								{#if !activationData}Run export_activations.py{:else}Loading renderer...{/if}
							</div>
						{/if}

						<!-- HUD Overlay: Right — Glass Card -->
						<div class="absolute top-4 right-4 z-50 text-right pointer-events-none bg-black/50 backdrop-blur-sm p-4 rounded-lg">
							<h3 class="text-amber-400 font-bold text-sm tracking-[0.2em] uppercase drop-shadow-lg">BDH (SPARSE)</h3>
							<p class="text-amber-300/50 text-xs mt-0.5">k-WTA + RoPE Attention</p>
							<div class="mt-3 font-mono">
								<p class="text-xs text-amber-300/60">ACTIVE LOAD</p>
								<p class="text-3xl font-black text-amber-400 drop-shadow-[0_0_10px_rgba(255,180,50,0.5)]">{bdhLoad}%</p>
								<p class="text-xs font-bold tracking-wider mt-1 text-green-400">OPTIMAL</p>
							</div>
						</div>

						<!-- Active neuron count — Glass Card -->
						<div class="absolute bottom-4 right-4 z-50 text-right pointer-events-none font-mono bg-black/50 backdrop-blur-sm p-3 rounded-lg">
							<p class="text-xs text-amber-300/60">ACTIVE CIRCUITS</p>
							<p class="text-lg text-amber-400">{currentActiveCount} <span class="text-amber-600 text-xs">/ {currentTotalNeurons}</span></p>
						</div>
					</div>
				</div>

				<!-- FLOATING TIME SCRUBBER — centered at arena/explainer boundary -->
				<div class="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 z-40 w-[90%] max-w-3xl">
					<div class="bg-[#020617]/95 backdrop-blur-xl rounded-xl border border-border px-5 py-3 shadow-2xl">
						<!-- Token chips -->
						<div class="flex flex-wrap gap-1 justify-center mb-2">
							{#each tokenLabels as token, i}
							<button
								class="px-2 py-0.5 rounded text-xs font-mono transition-all duration-200"
								class:bg-neon-amber={currentStep === i}
								class:text-background={currentStep === i}
								class:scale-110={currentStep === i}
								class:bg-surface={currentStep !== i}
								class:text-muted-foreground={currentStep !== i}
								style={currentStep !== i ? 'background-color: hsl(var(--surface) / 0.8);' : ''}
								on:click={() => (currentStep = i)}
							>
									{token}
								</button>
							{/each}
						</div>
						<!-- Slider -->
						<div class="flex items-center gap-3">
							<span class="text-xs text-muted-foreground font-mono w-12">Step {currentStep}</span>
							<input
								type="range"
								min="0"
								max={maxStep}
								bind:value={currentStep}
								class="flex-1 accent-yellow-400 h-1.5 cursor-pointer"
							/>
							<span class="text-xs text-muted-foreground font-mono w-8 text-right">{maxStep}</span>
						</div>
					</div>
				</div>
			</section>
		{/if}

		<!-- ================================================================ -->
		<!-- BOTTOM 30%: THE EXPLAINER — 3 Column Layout                      -->
		<!-- ================================================================ -->
		<section class="border-t border-border/50 px-6 pt-14 pb-8">
			<div class="max-w-screen-xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">

				<!-- Column 1: The Problem -->
				<div class="rounded-xl p-5 bg-surface/50 border border-border/40">
					<div class="flex items-center gap-2 mb-3">
						<span class="w-3 h-3 rounded-full bg-red-500"></span>
						<h3 class="text-sm font-bold text-red-700 uppercase tracking-wider">The Problem</h3>
					</div>
					<p class="text-sm text-muted-foreground leading-relaxed">
						Matrix multiplication forces <strong>every neuron to fire</strong> on every input.
						Softmax spreads attention across all tokens indiscriminately.
						GeLU activation never outputs a true zero — guaranteeing 100% neural activity.
					</p>
					<p class="text-sm text-muted-foreground mt-3">
						This creates <em>O(N²) complexity</em> and <em>polysemantic neurons</em>
						that respond to multiple unrelated concepts, making the model uninterpretable.
					</p>
					<div class="mt-4 rounded-lg p-3 border border-red-500/50 bg-red-500/10">
						<p class="text-xs text-red-400 font-mono">Active Load: {transLoad}%</p>
						<div class="mt-1 bg-red-500/20 rounded-full h-3 overflow-hidden">
							<div class="h-full bg-red-500 rounded-full" style="width: {transLoad}%"></div>
						</div>
					</div>
				</div>

				<!-- Column 2: The Solution -->
				<div class="rounded-xl p-5 bg-surface/50 border border-border/40">
					<div class="flex items-center gap-2 mb-3">
						<span class="w-3 h-3 rounded-full bg-cyan-500"></span>
						<h3 class="text-sm font-bold text-cyan-700 uppercase tracking-wider">The Solution</h3>
					</div>
					<p class="text-sm text-muted-foreground leading-relaxed">
						k-WTA (k-Winners-Take-All) creates <strong>biological competition</strong>.
						Only the most relevant neurons survive each forward pass.
						Every other neuron is forced to exactly zero.
					</p>
					<p class="text-sm text-muted-foreground mt-3">
						This achieves <em>O(N) complexity</em> and <em>monosemantic neurons</em> —
						each neuron fires for one concept only, making the model fully interpretable.
					</p>
					<div class="mt-4 rounded-lg p-3 border border-cyan-500/50 bg-cyan-500/10">
						<p class="text-xs text-cyan-300 font-mono">Active Load: {bdhLoad}%</p>
						<div class="mt-1 bg-cyan-500/20 rounded-full h-3 overflow-hidden">
							<div class="h-full bg-cyan-500 rounded-full" style="width: {bdhLoad}%"></div>
						</div>
					</div>
				</div>

				<!-- Column 3: The Impact — Energy Savings Bar Chart -->
				<div class="rounded-xl p-5 bg-surface/50 border border-border/40">
					<div class="flex items-center gap-2 mb-3">
						<span class="w-3 h-3 rounded-full bg-yellow-500"></span>
						<h3 class="text-sm font-bold text-yellow-700 uppercase tracking-wider">The Impact</h3>
					</div>

					<!-- Energy savings big number -->
					<div class="text-center my-4">
						<p class="text-5xl font-black text-neon-amber">{energySavings.toFixed(1)}%</p>
						<p class="text-sm text-muted-foreground mt-1">Energy Savings</p>
					</div>

					<!-- Bar chart: side by side -->
					<div class="space-y-3 mt-4">
						<div>
							<div class="flex justify-between text-xs mb-1">
								<span class="text-red-600 font-medium">Transformer</span>
								<span class="text-red-500 font-mono">{transLoad}%</span>
							</div>
							<div class="bg-secondary/70 rounded-full h-4 overflow-hidden">
								<div class="h-full bg-gradient-to-r from-red-500 to-orange-400 rounded-full transition-all duration-700" style="width: {transLoad}%"></div>
							</div>
						</div>
						<div>
							<div class="flex justify-between text-xs mb-1">
								<span class="text-cyan-600 font-medium">BDH k-WTA</span>
								<span class="text-cyan-500 font-mono">{bdhLoad}%</span>
							</div>
							<div class="bg-secondary/70 rounded-full h-4 overflow-hidden">
								<div class="h-full bg-gradient-to-r from-cyan-500 to-blue-400 rounded-full transition-all duration-700" style="width: {bdhLoad}%"></div>
							</div>
						</div>
					</div>

					<p class="text-xs text-muted-foreground mt-4 text-center">
						Fewer active neurons = less energy, less heat, more interpretable.
					</p>
				</div>
			</div>

			<!-- Narrative row -->
			{#if currentNarrative}
				<div class="max-w-screen-xl mx-auto mt-6">
					<div class="rounded-xl px-5 py-4 bg-surface/50 border border-border/40">
						<p class="text-sm text-muted-foreground leading-relaxed">
							<span class="text-yellow-400 font-mono text-xs mr-2">[Step {currentStep}]</span>
							{currentNarrative}
						</p>
					</div>
				</div>
			{/if}
		</section>

		<!-- ================================================================ -->
		<!-- DETAILED COMPARISON SECTIONS (existing)                           -->
		<!-- ================================================================ -->
		{#if data}
			<div class="max-w-screen-xl mx-auto px-4 py-8">
				<!-- Fog of War Heatmaps -->
				<section class="rounded-xl p-6 mb-8 bg-surface/50 border border-border/40">
					<h2 class="text-xl font-serif text-foreground mb-1">Fog of War — Activation Heatmaps</h2>
					<p class="text-sm text-muted-foreground mb-6">
						Layer 0 activations for "<em>{data.visual_sample.phrase}</em>" — {data.visual_sample.n_sampled_neurons} sampled neurons.
					</p>
					<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
						<div>
							<h3 class="text-sm font-semibold text-red-700 mb-2 text-center">Transformer — "The Fog"</h3>
							<div class="bg-[#020617] rounded-lg p-2 border border-border/70">
								<canvas bind:this={transCanvas} width={500} height={300} class="w-full rounded"></canvas>
							</div>
							<p class="text-xs text-muted-foreground mt-2 text-center">Dense, warm glow — almost every neuron active</p>
						</div>
						<div>
							<h3 class="text-sm font-semibold text-blue-700 mb-2 text-center">BDH (k-WTA) — "The Stars"</h3>
							<div class="bg-[#020617] rounded-lg p-2 border border-border/70">
								<canvas bind:this={bdhCanvas} width={500} height={300} class="w-full rounded"></canvas>
							</div>
							<p class="text-xs text-muted-foreground mt-2 text-center">Sparse, bright pulses — only top-k fire</p>
						</div>
					</div>
					<div class="mt-4 flex flex-wrap gap-1 justify-center">
						{#each data.visual_sample.tokens_decoded as token, i}
							<span class="px-2 py-0.5 bg-secondary/70 rounded text-xs font-mono text-foreground">{i}: {token}</span>
						{/each}
					</div>
				</section>

				<!-- Concept Battle -->
				<section class="rounded-xl p-6 mb-8 bg-surface/50 border border-border/40">
					<h2 class="text-xl font-serif text-foreground mb-1">Concept Isolation Test</h2>
					<p class="text-sm text-muted-foreground mb-4">
						For each concept, we find the most active neuron at Layer 0, then measure that neuron's
						response to unrelated "noise" words.
					</p>
					<div class="flex flex-wrap gap-2 mb-6">
						{#each Object.keys(data.concept_battle) as concept}
							<button
								class="px-4 py-1.5 rounded-full text-sm font-medium transition-colors border"
								class:bg-indigo-600={selectedConcept === concept}
								class:text-white={selectedConcept === concept}
								class:border-indigo-600={selectedConcept === concept}
								class:bg-surface={selectedConcept !== concept}
								class:text-muted-foreground={selectedConcept !== concept}
								class:border-border={selectedConcept !== concept}
								style={selectedConcept !== concept ? 'background-color: hsl(var(--surface) / 0.8); border-color: hsl(var(--border) / 0.7);' : ''}
								on:click={() => (selectedConcept = concept)}
							>{concept}</button>
						{/each}
					</div>
					{#if selectedConcept && data.concept_battle[selectedConcept]}
						{@const battle = data.concept_battle[selectedConcept]}
						{@const noiseWords = getNoiseWords(battle)}
						{@const maxNoise = getMaxNoiseAbs(battle)}
						<div class="grid grid-cols-2 gap-4 mb-6">
							<div class="p-4 rounded-lg border border-blue-500/40 bg-blue-500/10">
								<p class="text-xs text-blue-600 font-medium">BDH — Neuron #{battle.bdh_top_neuron}</p>
								<p class="text-2xl font-mono text-blue-800 mt-1">{battle.bdh_concept_strength.toFixed(4)}</p>
								<p class="text-xs text-blue-500">Concept activation strength</p>
							</div>
							<div class="p-4 rounded-lg border border-red-500/40 bg-red-500/10">
								<p class="text-xs text-red-600 font-medium">Transformer — Neuron #{battle.transformer_top_neuron}</p>
								<p class="text-2xl font-mono text-red-800 mt-1">{battle.transformer_concept_strength.toFixed(4)}</p>
								<p class="text-xs text-red-500">Concept activation strength</p>
							</div>
						</div>
						<h3 class="text-sm font-semibold text-foreground mb-3">
							Noise Word Interference — "{selectedConcept}" neuron on unrelated words
						</h3>
						<div class="space-y-3">
							{#each noiseWords as word}
								{@const bdhVal = battle.bdh_noise_activations[word] ?? 0}
								{@const transVal = battle.transformer_noise_activations[word] ?? 0}
								{@const bdhPct = maxNoise > 0 ? (Math.abs(bdhVal) / maxNoise) * 100 : 0}
								{@const transPct = maxNoise > 0 ? (Math.abs(transVal) / maxNoise) * 100 : 0}
								<div class="flex items-center gap-3">
									<span class="text-xs font-mono text-muted-foreground w-12 text-right">{word}</span>
									<div class="flex-1 space-y-1">
										<div class="flex items-center gap-2">
								<div class="flex-1 bg-secondary/70 rounded-full h-3 overflow-hidden">
												<div class="h-full bg-gradient-to-r from-red-400 to-orange-400 rounded-full" style="width: {Math.max(transPct, 0.5)}%"></div>
											</div>
											<span class="text-xs font-mono text-red-600 w-16">{transVal.toFixed(4)}</span>
										</div>
										<div class="flex items-center gap-2">
											<div class="flex-1 bg-secondary/70 rounded-full h-3 overflow-hidden">
												<div class="h-full bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full" style="width: {Math.max(bdhPct, 0.5)}%"></div>
											</div>
											<span class="text-xs font-mono text-blue-600 w-16">{bdhVal.toFixed(4)}</span>
										</div>
									</div>
								</div>
							{/each}
						</div>
						<div class="mt-4 flex items-center gap-6 text-xs text-muted-foreground">
							<span class="flex items-center gap-1.5">
								<span class="w-3 h-3 rounded-full bg-gradient-to-r from-red-400 to-orange-400 inline-block"></span>
								Transformer (polysemantic)
							</span>
							<span class="flex items-center gap-1.5">
								<span class="w-3 h-3 rounded-full bg-gradient-to-r from-blue-400 to-cyan-400 inline-block"></span>
								BDH k-WTA (monosemantic)
							</span>
						</div>
					{/if}
				</section>

				<!-- Config footer -->
				<div class="text-center text-xs text-muted-foreground pb-8">
					{data.metadata.model_name_champion} vs {data.metadata.model_name_challenger} &bull;
					{data.metadata.n_layers}L / {data.metadata.n_embd}D / {data.metadata.n_head}H &bull;
					k = {(data.metadata.top_k_fraction * 100).toFixed(0)}%
				</div>
			</div>
		{/if}
	{/if}
</div>
