<script lang="ts">
	import { onMount } from 'svelte';

	const API_URL = 'http://localhost:8000';

	interface SparsePulse {
		token: string;
		step: number;
		neuron_id: number;
		activation: number;
	}

	interface AtlasEntry {
		concept: string;
		triggers: string[];
		score: number;
	}

	let tokens: string[] = [];
	let pulses: SparsePulse[] = [];
	let atlas: Record<string, AtlasEntry> = {};
	let isLoading = true;
	let error = '';
	let selectedStep = 0;
	let viewMode: 'radial' | 'scatter' | 'stream' = 'radial';
	let showAtlasLabels = true;

	async function loadData() {
		try {
			const [brainRes, atlasRes] = await Promise.all([
				fetch(`${API_URL}/api/sparse-brain`),
				fetch(`${API_URL}/api/activation-atlas`),
			]);
			if (!brainRes.ok) throw new Error(`Brain API error: ${brainRes.status}`);
			if (!atlasRes.ok) throw new Error(`Atlas API error: ${atlasRes.status}`);

			const brainData = await brainRes.json();
			const atlasData = await atlasRes.json();

			tokens = brainData.tokens;
			pulses = brainData.activations;
			atlas = atlasData.atlas;
			isLoading = false;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load data';
			isLoading = false;
		}
	}

	onMount(loadData);

	// Get pulses for current step
	$: stepPulses = pulses.filter((p) => p.step === selectedStep);
	$: maxActivation = Math.max(...stepPulses.map((p) => p.activation), 0.1);

	// All unique neuron IDs for layout
	$: allNeuronIds = [...new Set(pulses.map((p) => p.neuron_id))].sort((a, b) => a - b);
	$: maxNeuronId = Math.max(...allNeuronIds, 1);

	// Get concept label for a neuron
	function getConceptLabel(neuronId: number): string {
		const entry = atlas[String(neuronId)];
		return entry ? entry.concept : '';
	}

	// Concept-based coloring
	const conceptColorMap: Record<string, string> = {};
	const colorPalette = [
		'#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
		'#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#a855f7',
		'#6366f1', '#0ea5e9', '#84cc16', '#f59e0b', '#d946ef',
	];

	function getNeuronColor(neuronId: number): string {
		const concept = getConceptLabel(neuronId);
		if (!concept) return '#64748b';
		if (!conceptColorMap[concept]) {
			conceptColorMap[concept] = colorPalette[Object.keys(conceptColorMap).length % colorPalette.length];
		}
		return conceptColorMap[concept];
	}

	// Per-token stats
	$: tokenStats = tokens.map((_, step) => {
		const sp = pulses.filter((p) => p.step === step);
		return {
			count: sp.length,
			meanAct: sp.length > 0 ? sp.reduce((s, p) => s + p.activation, 0) / sp.length : 0,
			maxAct: sp.length > 0 ? Math.max(...sp.map((p) => p.activation)) : 0,
		};
	});
</script>

<svelte:head>
	<title>Monosemanticity Visualization - BDH</title>
</svelte:head>

<div class="max-w-screen-2xl mx-auto px-4 py-6 min-h-[calc(100vh-4rem)] flex flex-col">
	<div class="mb-4">
		<a href="/" class="text-sm text-primary hover:text-primary/80 mb-2 inline-block">
			← Back to Visualization
		</a>
		<h1 class="text-2xl font-serif text-foreground mb-1">Monosemanticity Visualization</h1>
		<p class="text-sm text-muted-foreground">
			Explore which neurons fire for each token. Each dot is a neuron; color encodes its semantic concept.
			Size and brightness reflect activation strength.
		</p>
	</div>

	{#if isLoading}
		<div class="flex items-center justify-center h-[60vh]">
			<div class="text-center">
				<div class="spinner mx-auto mb-4"></div>
				<p class="text-muted-foreground">Loading sparse brain data...</p>
			</div>
		</div>
	{:else if error}
		<div class="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
			<p class="text-red-700 mb-2">{error}</p>
			<p class="text-sm text-red-500">Make sure the backend is running: <code class="bg-red-100 px-2 py-1 rounded">python main.py</code></p>
		</div>
	{:else}
		<!-- Controls -->
		<div class="flex items-center gap-4 mb-3">
			<div class="flex gap-1 bg-surface rounded-lg p-1 border border-border/60">
				<button class="px-3 py-1 text-sm rounded-md transition-colors text-muted-foreground" class:bg-surface-hover={viewMode === 'radial'} class:text-foreground={viewMode === 'radial'} class:shadow-sm={viewMode === 'radial'} on:click={() => (viewMode = 'radial')}>
					Radial Map
				</button>
				<button class="px-3 py-1 text-sm rounded-md transition-colors text-muted-foreground" class:bg-surface-hover={viewMode === 'scatter'} class:text-foreground={viewMode === 'scatter'} class:shadow-sm={viewMode === 'scatter'} on:click={() => (viewMode = 'scatter')}>
					Scatter
				</button>
				<button class="px-3 py-1 text-sm rounded-md transition-colors text-muted-foreground" class:bg-surface-hover={viewMode === 'stream'} class:text-foreground={viewMode === 'stream'} class:shadow-sm={viewMode === 'stream'} on:click={() => (viewMode = 'stream')}>
					Stream
				</button>
			</div>
			<label class="flex items-center gap-2 text-sm">
				<input type="checkbox" bind:checked={showAtlasLabels} class="accent-primary" />
				Concept labels
			</label>
			<span class="text-xs text-muted-foreground ml-auto">
				Token: <span class="font-mono font-medium">{tokens[selectedStep]?.trim() || '_'}</span>
				| Active neurons: <span class="font-mono">{stepPulses.length}</span>
			</span>
		</div>

		<div class="grid lg:grid-cols-3 gap-6 flex-1 items-stretch">
			<!-- Main Visualization -->
			<div class="lg:col-span-2 flex flex-col gap-4">
				<!-- Bare viewport without extra card box -->
				<div class="rounded-xl overflow-hidden">
					{#if viewMode === 'radial'}
						<!-- Radial Neuron Map -->
						<svg
							viewBox="0 0 700 700"
							class="w-full"
							style="
								height: 550px;
								background:
									radial-gradient(circle at top, rgba(56,189,248,0.08), transparent 55%),
									radial-gradient(circle at bottom, rgba(244,114,182,0.08), transparent 55%),
									linear-gradient(180deg, #020617 0%, #020314 100%);
							"
						>
							<!-- Concentric rings -->
							{#each [100, 200, 280] as r}
								<circle cx="350" cy="350" {r} fill="none" stroke="#334155" stroke-width="0.5" opacity="0.3" />
							{/each}

							<!-- Neuron positions: arrange by neuron_id angle, activation radius -->
							{#each stepPulses as pulse}
								{@const angle = (pulse.neuron_id / maxNeuronId) * 2 * Math.PI - Math.PI / 2}
								{@const radius = 80 + (pulse.activation / maxActivation) * 220}
								{@const x = 350 + Math.cos(angle) * radius}
								{@const y = 350 + Math.sin(angle) * radius}
								{@const dotSize = 3 + (pulse.activation / maxActivation) * 10}
								{@const color = getNeuronColor(pulse.neuron_id)}

								<!-- Glow -->
								<circle
									cx={x}
									cy={y}
									r={dotSize * 2.5}
									fill={color}
									opacity={0.1 + (pulse.activation / maxActivation) * 0.15}
								/>

								<!-- Core -->
								<circle
									cx={x}
									cy={y}
									r={dotSize}
									fill={color}
									opacity={0.5 + (pulse.activation / maxActivation) * 0.5}
								>
									<title>N{pulse.neuron_id} | {getConceptLabel(pulse.neuron_id) || '?'} | act: {pulse.activation.toFixed(4)}</title>
								</circle>

								<!-- Label -->
								{#if showAtlasLabels && dotSize > 8}
									{@const label = getConceptLabel(pulse.neuron_id)}
									{#if label}
										<text
											x={x}
											y={y - dotSize - 4}
											text-anchor="middle"
											font-size="8"
											fill={color}
											opacity="0.8"
										>
											{label}
										</text>
									{/if}
								{/if}
							{/each}

							<!-- Center: current token -->
							<circle cx="350" cy="350" r="35" fill="#1e293b" stroke="#475569" stroke-width="1" />
							<text x="350" y="346" text-anchor="middle" font-size="18" fill="#e2e8f0" font-family="monospace">
								{tokens[selectedStep]?.trim() || '_'}
							</text>
							<text x="350" y="365" text-anchor="middle" font-size="9" fill="#64748b">
								token {selectedStep}
							</text>
						</svg>
					{:else if viewMode === 'scatter'}
						<!-- Scatter: neuron_id vs activation -->
						<svg
							viewBox="0 0 700 500"
							class="w-full"
							style="
								height: 500px;
								background:
									radial-gradient(circle at top, rgba(56,189,248,0.06), transparent 55%),
									radial-gradient(circle at bottom, rgba(244,114,182,0.06), transparent 55%),
									linear-gradient(180deg, #020617 0%, #020314 100%);
							"
						>
							<!-- Axes -->
							<line x1="60" y1="440" x2="680" y2="440" stroke="#cbd5e1" stroke-width="1" />
							<line x1="60" y1="20" x2="60" y2="440" stroke="#cbd5e1" stroke-width="1" />
							<text x="370" y="475" text-anchor="middle" font-size="12" fill="#64748b">Neuron ID</text>
							<text x="15" y="230" text-anchor="middle" font-size="12" fill="#64748b" transform="rotate(-90, 15, 230)">Activation</text>

							<!-- Grid lines -->
							{#each [0.25, 0.5, 0.75, 1.0] as tick}
								{@const y = 440 - tick * 400}
								<line x1="60" y1={y} x2="680" y2={y} stroke="#e2e8f0" stroke-width="0.5" />
								<text x="55" y={y + 4} text-anchor="end" font-size="9" fill="#94a3b8" font-family="monospace">
									{(tick * maxActivation).toFixed(2)}
								</text>
							{/each}

							<!-- Data points -->
							{#each stepPulses as pulse}
								{@const x = 60 + (pulse.neuron_id / maxNeuronId) * 620}
								{@const y = 440 - (pulse.activation / maxActivation) * 400}
								{@const color = getNeuronColor(pulse.neuron_id)}
								{@const r = 3 + (pulse.activation / maxActivation) * 6}

								<circle
									cx={x}
									cy={y}
									{r}
									fill={color}
									opacity="0.7"
								>
									<title>N{pulse.neuron_id} | {getConceptLabel(pulse.neuron_id) || '?'} | act: {pulse.activation.toFixed(4)}</title>
								</circle>
							{/each}
						</svg>
					{:else}
						<!-- Stream: all tokens over time -->
						<svg
							viewBox="0 0 700 500"
							class="w-full"
							style="
								height: 500px;
								background:
									radial-gradient(circle at top, rgba(56,189,248,0.06), transparent 55%),
									radial-gradient(circle at bottom, rgba(244,114,182,0.06), transparent 55%),
									linear-gradient(180deg, #020617 0%, #020314 100%);
							"
						>
							<!-- Token columns -->
							{#each tokens as token, step}
								{@const colWidth = 700 / tokens.length}
								{@const x0 = step * colWidth}

								<!-- Column highlight for selected -->
								{#if step === selectedStep}
									<rect x={x0} y="0" width={colWidth} height="500" fill="#3b82f6" opacity="0.08" />
								{/if}

								<!-- Token label at bottom -->
								<text
									x={x0 + colWidth / 2}
									y="490"
									text-anchor="middle"
									font-size={Math.min(9, colWidth - 1)}
									fill={step === selectedStep ? '#60a5fa' : '#475569'}
									font-family="monospace"
								>
									{token.trim() || '_'}
								</text>

								<!-- Neuron dots in column -->
								{#each pulses.filter((p) => p.step === step) as pulse}
									{@const yPos = 10 + (pulse.neuron_id / maxNeuronId) * 460}
									{@const dotR = 1 + (pulse.activation / maxActivation) * 3}
									{@const color = getNeuronColor(pulse.neuron_id)}

									<circle
										cx={x0 + colWidth / 2}
										cy={yPos}
										r={dotR}
										fill={color}
										opacity={0.4 + (pulse.activation / maxActivation) * 0.6}
									>
										<title>{token}: N{pulse.neuron_id} | {getConceptLabel(pulse.neuron_id)} | {pulse.activation.toFixed(4)}</title>
									</circle>
								{/each}
							{/each}

							<!-- Y-axis label -->
							<text x="8" y="250" text-anchor="middle" font-size="10" fill="#475569" transform="rotate(-90, 8, 250)">
								Neuron ID
							</text>
						</svg>
					{/if}
				</div>

				<!-- Token scrubber -->
				<div class="rounded-xl p-4 bg-surface/60 border border-border/40">
					<div class="flex items-center gap-4">
						<button
							on:click={() => (selectedStep = Math.max(0, selectedStep - 1))}
							class="p-2 rounded hover:bg-surface-hover/80"
							disabled={selectedStep === 0}
						>
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
							</svg>
						</button>
						<input
							type="range"
							min="0"
							max={tokens.length - 1}
							bind:value={selectedStep}
							class="mono-slider flex-1"
						/>
						<button
							on:click={() => (selectedStep = Math.min(tokens.length - 1, selectedStep + 1))}
							class="p-2 rounded hover:bg-surface-hover/80"
							disabled={selectedStep === tokens.length - 1}
						>
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
							</svg>
						</button>
						<span class="font-mono text-xs text-muted-foreground">{selectedStep + 1}/{tokens.length}</span>
					</div>

					<!-- Token chips -->
					<div class="flex flex-wrap gap-1 mt-3">
						{#each tokens as token, i}
							<button
								class="px-2 py-0.5 text-xs font-mono rounded transition-all ring-0 shadow-none border-0"
								class:bg-primary={i === selectedStep}
								class:text-primary-foreground={i === selectedStep}
								class:bg-surface-hover={i !== selectedStep}
								on:click={() => (selectedStep = i)}
							>
								{token.trim() || '_'}
							</button>
						{/each}
					</div>
				</div>
			</div>

			<!-- Right panel -->
			<div class="flex flex-col gap-4 h-full">
				<!-- Active neurons list -->
				<div class="rounded-xl p-4 bg-surface/60 border border-border/40 flex-1 min-h-0 overflow-hidden flex flex-col">
					<h3 class="font-serif text-sm mb-2 text-muted-foreground uppercase tracking-wider flex-shrink-0">Active Neurons</h3>
					<div class="min-h-[120px] max-h-[280px] overflow-y-auto flex-1">
					{#if stepPulses.length === 0}
						<p class="text-xs text-muted-foreground">No active neurons at this step</p>
					{:else}
						<div class="flex flex-col gap-1.5">
							{#each stepPulses.sort((a, b) => b.activation - a.activation).slice(0, 30) as pulse}
								{@const concept = getConceptLabel(pulse.neuron_id)}
								{@const color = getNeuronColor(pulse.neuron_id)}
								<div class="grid grid-cols-[auto_3rem_1fr_4rem_3rem] items-center gap-2 py-1.5 px-2 rounded hover:bg-surface-hover/70 min-h-[28px]">
									<div class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: {color}"></div>
									<span class="font-mono text-xs truncate">N{pulse.neuron_id}</span>
									<span class="text-xs truncate px-1.5 py-0.5 rounded min-w-0" style="background-color: {color}20; color: {color}">
										{concept || '—'}
									</span>
									<div class="h-1.5 bg-secondary/70 rounded-full overflow-hidden min-w-0">
										<div
											class="h-full rounded-full"
											style="width: {(pulse.activation / maxActivation) * 100}%; background-color: {color}"
										></div>
									</div>
									<span class="font-mono text-xs text-muted-foreground text-right truncate">
										{pulse.activation.toFixed(3)}
									</span>
								</div>
							{/each}
						</div>
					{/if}
					</div>
				</div>

				<!-- Token activity sparkline -->
				<div class="rounded-xl p-4 bg-surface/50 border border-border/40">
					<h3 class="font-serif text-sm mb-2 text-muted-foreground uppercase tracking-wider">Activity per Token</h3>
					<svg viewBox="0 0 300 80" class="w-full">
						{#each tokenStats as stat, i}
							{@const barH = (stat.count / Math.max(...tokenStats.map((s) => s.count), 1)) * 60}
							{@const x = (i / tokens.length) * 280 + 10}
							<rect
								{x}
								y={70 - barH}
								width={Math.max(1, 280 / tokens.length - 1)}
								height={barH}
								fill={i === selectedStep ? '#00d4ff' : '#1e293b'}
								opacity={i === selectedStep ? 1 : 0.6}
								rx="1"
								stroke="none"
								class="activity-bar cursor-pointer"
								on:click={() => (selectedStep = i)}
								role="button"
								tabindex="0"
								on:keydown={(e) => e.key === 'Enter' && (selectedStep = i)}
							>
								<title>{tokens[i]}: {stat.count} neurons, max: {stat.maxAct.toFixed(3)}</title>
							</rect>
						{/each}
						<line x1="10" y1="70" x2="290" y2="70" stroke="#475569" stroke-width="0.5" opacity="0.6" />
					</svg>
				</div>

				<!-- Concept color legend -->
				<div class="rounded-xl p-4 bg-surface/60 border border-border/40">
					<h3 class="font-serif text-sm mb-2 text-muted-foreground uppercase tracking-wider">Concept Colors</h3>
					<div class="flex flex-wrap gap-2">
						{#each Object.entries(conceptColorMap).slice(0, 15) as [concept, color]}
							<div class="flex items-center gap-1.5">
								<div class="w-2.5 h-2.5 rounded-full" style="background-color: {color}"></div>
								<span class="text-xs capitalize">{concept}</span>
							</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.spinner {
		@apply w-8 h-8 border-2 border-gray-200 rounded-full;
		border-top-color: hsl(var(--primary));
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	/* Slider: blend with dark background */
	.mono-slider {
		-webkit-appearance: none;
		appearance: none;
		height: 6px;
		background: hsl(var(--secondary));
		border-radius: 3px;
		outline: none;
	}
	.mono-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: hsl(var(--primary));
		border: 1px solid hsl(var(--border));
		cursor: pointer;
		box-shadow: 0 0 8px hsl(var(--primary) / 0.4);
	}
	.mono-slider::-moz-range-thumb {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: hsl(var(--primary));
		border: 1px solid hsl(var(--border));
		cursor: pointer;
		box-shadow: 0 0 8px hsl(var(--primary) / 0.4);
	}

	/* Activity per Token bars: no border, stroke, or focus ring */
	.activity-bar,
	.activity-bar:focus,
	.activity-bar:focus-visible {
		outline: none;
		stroke: none;
		box-shadow: none;
	}
</style>
