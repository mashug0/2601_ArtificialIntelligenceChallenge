<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	const API_URL = '';

	interface Frame {
		step: number;
		current_token: string;
		matrix: number[][];
	}

	let tokens: string[] = [];
	let frames: Frame[] = [];
	let currentFrame = 0;
	let isLoading = true;
	let isPlaying = false;
	let error = '';
	let animationFrame: number;
	let lastTickTime = 0;
	let speed = 800; // ms per frame
	let showLabels = true;

	async function loadData() {
		try {
			const res = await fetch(`${API_URL}/api/hebbian-trace`);
			if (!res.ok) throw new Error(`API error: ${res.status}`);
			const data = await res.json();
			tokens = data.tokens;
			frames = data.frames;
			isLoading = false;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load hebbian trace';
			isLoading = false;
		}
	}

	onMount(loadData);

	function play() {
		isPlaying = true;
		lastTickTime = performance.now();
		tick(lastTickTime);
	}

	function pause() {
		isPlaying = false;
		if (animationFrame) cancelAnimationFrame(animationFrame);
	}

	function tick(now: number) {
		if (!isPlaying) return;
		const elapsed = now - lastTickTime;
		if (elapsed >= speed) {
			lastTickTime = now;
			if (currentFrame < frames.length - 1) {
				currentFrame++;
			} else {
				currentFrame = 0;
			}
		}
		animationFrame = requestAnimationFrame(tick);
	}

	onDestroy(() => {
		if (animationFrame) cancelAnimationFrame(animationFrame);
	});

	// Get color for attention weight
	function getColor(value: number): string {
		if (value < 0.01) return 'rgba(15, 23, 42, 0.3)';
		const intensity = Math.min(value * 2, 1);
		// Blue-green-yellow gradient
		if (intensity < 0.33) {
			const t = intensity / 0.33;
			return `rgba(59, ${Math.round(130 + t * 70)}, ${Math.round(246 - t * 100)}, ${0.4 + intensity * 0.6})`;
		} else if (intensity < 0.66) {
			const t = (intensity - 0.33) / 0.33;
			return `rgba(${Math.round(59 + t * 190)}, ${Math.round(200 + t * 55)}, ${Math.round(146 - t * 100)}, ${0.7 + intensity * 0.3})`;
		} else {
			const t = (intensity - 0.66) / 0.34;
			return `rgba(${Math.round(249)}, ${Math.round(255 - t * 60)}, ${Math.round(46 + t * 30)}, 1)`;
		}
	}

	$: frame = frames[currentFrame];
	$: matrixSize = frame ? frame.matrix.length : 0;
	// Increased cell size to make matrix larger - max 40px, base width 800px
	$: cellSize = matrixSize > 0 ? Math.min(40, 800 / matrixSize) : 40;
</script>

<svelte:head>
	<title>Hebbian Synapse Matrix - BDH Visualization</title>
</svelte:head>

<div class="max-w-screen-2xl mx-auto px-4 py-6">
	<div class="mb-6">
		<a href="/" class="text-xs text-primary hover:text-primary/80 mb-2 inline-block font-mono tracking-[0.18em] uppercase">
			← Back to Visualization
		</a>
		<h1 class="text-3xl font-serif text-paper-text mb-2">Hebbian Synapse Matrix</h1>
		<p class="text-paper-muted">
			Watch how attention synapses strengthen as each token is processed.
			Brighter cells indicate stronger connections - "neurons that fire together, wire together."
		</p>
	</div>

	{#if isLoading}
		<div class="flex items-center justify-center h-[60vh]">
			<div class="text-center">
				<div class="spinner mx-auto mb-4"></div>
				<p class="text-paper-muted">Loading hebbian trace...</p>
			</div>
		</div>
	{:else if error}
		<div class="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
			<p class="text-red-700 mb-2">{error}</p>
			<p class="text-sm text-red-500">Make sure the backend is running: <code class="bg-red-100 px-2 py-1 rounded">python main.py</code></p>
		</div>
	{:else if frame}
		<div class="grid lg:grid-cols-3 gap-6">
			<!-- Matrix Visualization -->
			<div class="lg:col-span-2">
				<!-- Softer viewport without prominent box -->
				<div class="rounded-xl p-4">
					<div class="flex items-center justify-between mb-4">
						<h2 class="font-serif text-lg">Attention Matrix (Head 0, Layer 0)</h2>
						<div class="flex items-center gap-2">
							<span class="text-sm text-paper-muted">Step {currentFrame + 1}/{frames.length}</span>
								<span class="px-2 py-0.5 rounded text-xs font-mono bg-primary/15 text-primary border border-primary/40">
								{frame.current_token}
							</span>
						</div>
					</div>

					<!-- SVG Matrix -->
					<div class="overflow-auto min-h-[600px] flex items-center justify-center">
						<svg
							width={matrixSize * cellSize + 80}
							height={matrixSize * cellSize + 80}
							class="mx-auto"
							style="
								background:
									radial-gradient(circle at top, rgba(56,189,248,0.06), transparent 60%),
									radial-gradient(circle at bottom, rgba(244,114,182,0.06), transparent 60%),
									linear-gradient(180deg, #020617 0%, #020314 100%);
								border-radius: 0.75rem;
							"
						>
							<!-- Column headers (keys) -->
							{#if showLabels}
								{#each tokens as token, i}
									<text
										x={60 + i * cellSize + cellSize / 2}
										y={45}
										text-anchor="middle"
										font-size={Math.min(10, cellSize - 2)}
										fill={i < frame.step ? '#334155' : '#cbd5e1'}
										font-family="monospace"
										transform="rotate(-45, {60 + i * cellSize + cellSize / 2}, {45})"
									>
										{token.trim() || '_'}
									</text>
								{/each}
							{/if}

							<!-- Row headers (queries) -->
							{#if showLabels}
								{#each tokens as token, i}
									<text
										x={55}
										y={60 + i * cellSize + cellSize / 2 + 4}
										text-anchor="end"
										font-size={Math.min(10, cellSize - 2)}
										fill={i < frame.step ? '#334155' : '#cbd5e1'}
										font-family="monospace"
									>
										{token.trim() || '_'}
									</text>
								{/each}
							{/if}

							<!-- Matrix cells -->
							{#each frame.matrix as row, i}
								{#each row as value, j}
									<rect
										x={60 + j * cellSize}
										y={60 + i * cellSize}
										width={cellSize - 1}
										height={cellSize - 1}
										fill={i < frame.step && j < frame.step ? getColor(value) : 'rgba(15, 23, 42, 0.05)'}
										rx="2"
									>
										<title>Q:{tokens[i]} → K:{tokens[j]} = {value.toFixed(4)}</title>
									</rect>
								{/each}
							{/each}

							<!-- Active row/col highlight -->
							{#if frame.step > 0}
								<rect
									x={60}
									y={60 + (frame.step - 1) * cellSize}
									width={frame.step * cellSize}
									height={cellSize - 1}
									fill="none"
									stroke="#f59e0b"
									stroke-width="2"
									rx="2"
								/>
								<rect
									x={60 + (frame.step - 1) * cellSize}
									y={60}
									width={cellSize - 1}
									height={frame.step * cellSize}
									fill="none"
									stroke="#f59e0b"
									stroke-width="2"
									rx="2"
								/>
							{/if}

							<!-- Axis labels -->
							<text x={60 + matrixSize * cellSize / 2} y={60 + matrixSize * cellSize + 20} text-anchor="middle" font-size="12" fill="#64748b">Keys (attended to)</text>
							<text x={15} y={60 + matrixSize * cellSize / 2} text-anchor="middle" font-size="12" fill="#64748b" transform="rotate(-90, 15, {60 + matrixSize * cellSize / 2})">Queries (attending)</text>
						</svg>
					</div>

					<!-- Color scale legend -->
					<div class="flex items-center gap-4 mt-4 justify-center">
						<span class="text-xs text-paper-muted">Weak</span>
						<div class="flex h-3 rounded overflow-hidden" style="width: 200px;">
							{#each Array(20) as _, i}
								<div style="flex:1; background-color: {getColor(i / 20)}"></div>
							{/each}
						</div>
						<span class="text-xs text-paper-muted">Strong</span>
					</div>
				</div>
			</div>

			<!-- Controls & Info -->
			<div class="space-y-4">
				<!-- Playback -->
				<div class="rounded-xl p-4 bg-surface/50 border border-border/40">
					<h3 class="font-serif text-sm mb-3 text-muted-foreground uppercase tracking-wider">Playback</h3>

					<div class="flex gap-2 mb-4">
						{#if isPlaying}
							<button
								on:click={pause}
								class="flex-1 px-4 py-2 rounded-md bg-red-600 text-white hover:bg-red-500 transition-colors text-xs font-mono"
							>
								Pause
							</button>
						{:else}
							<button
								on:click={play}
								class="flex-1 px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors text-xs font-mono"
							>
								Play
							</button>
						{/if}
						<button
							on:click={() => (currentFrame = 0)}
							class="px-4 py-2 rounded-md bg-surface text-muted-foreground border border-border hover:bg-surface-hover transition-colors text-xs font-mono"
						>
							Reset
						</button>
					</div>

					<div class="mb-3">
						<label class="text-xs text-muted-foreground mb-1 block" for="frame-slider">Frame</label>
						<input
							id="frame-slider"
							type="range"
							min="0"
							max={frames.length - 1}
							bind:value={currentFrame}
							class="hebbian-slider w-full"
						/>
					</div>

					<div class="mb-3">
						<label class="text-xs text-muted-foreground mb-1 block" for="speed-slider">Speed: {speed}ms</label>
						<input
							id="speed-slider"
							type="range"
							min="200"
							max="2000"
							step="100"
							bind:value={speed}
							class="hebbian-slider w-full"
						/>
					</div>

					<label class="flex items-center gap-2 text-xs text-muted-foreground">
						<input type="checkbox" bind:checked={showLabels} class="accent-primary" />
						Show token labels
					</label>
				</div>

				<!-- Current state -->
				<div class="rounded-xl p-4 bg-surface/50 border border-border/40">
					<h3 class="font-serif text-sm mb-3 text-muted-foreground uppercase tracking-wider">Current State</h3>
					<div class="space-y-2 text-sm">
						<div class="flex justify-between">
							<span class="text-muted-foreground">Current Token</span>
							<span class="font-mono px-2 py-0.5 rounded bg-primary/15 text-primary border border-primary/40">
								{frame.current_token}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-muted-foreground">Tokens Processed</span>
							<span class="font-mono text-foreground">{frame.step}/{tokens.length}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-muted-foreground">Matrix Coverage</span>
							<span class="font-mono text-foreground">{((frame.step / tokens.length) * 100).toFixed(0)}%</span>
						</div>
					</div>
				</div>

				<!-- Token sequence -->
				<div class="rounded-xl p-4 bg-surface/50 border border-border/40">
					<h3 class="font-serif text-sm mb-3 text-muted-foreground uppercase tracking-wider">Token Sequence</h3>
					<div class="flex flex-wrap gap-1.5">
						{#each tokens as token, i}
							<button
								class="token-chip px-2.5 py-1 text-xs font-mono rounded-full border transition-all duration-200"
								class:token-current={i === frame.step - 1}
								class:token-past={i < frame.step - 1}
								class:token-future={i >= frame.step}
								on:click={() => (currentFrame = i)}
							>
								{token.trim() || '_'}
							</button>
						{/each}
					</div>
				</div>

				<!-- Explanation -->
				<div class="rounded-xl p-4 bg-surface/50 border border-border/40">
					<h3 class="font-serif text-sm mb-2 text-muted-foreground uppercase tracking-wider">How Hebbian Learning Works</h3>
					<p class="text-sm text-muted-foreground mb-2 leading-relaxed">
						The attention matrix shows which tokens attend to which other tokens.
						As more tokens are processed, the triangular matrix grows — causal masking.
					</p>
					<p class="text-sm text-muted-foreground leading-relaxed">
						Bright cells = strong synaptic connections, mirroring Hebbian plasticity.
					</p>
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.spinner {
		@apply w-8 h-8 border-2 border-gray-200 rounded-full;
		border-top-color: #2563eb;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	/* Sliders: blend with dark background */
	.hebbian-slider {
		-webkit-appearance: none;
		appearance: none;
		height: 6px;
		background: hsl(var(--secondary));
		border-radius: 3px;
		outline: none;
	}
	.hebbian-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: hsl(var(--primary));
		border: 1px solid hsl(var(--border));
		cursor: pointer;
		box-shadow: 0 0 8px hsl(var(--primary) / 0.4);
	}
	.hebbian-slider::-moz-range-thumb {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: hsl(var(--primary));
		border: 1px solid hsl(var(--border));
		cursor: pointer;
		box-shadow: 0 0 8px hsl(var(--primary) / 0.4);
	}

	/* Token sequence chips */
	.token-chip {
		background: hsl(var(--secondary) / 0.6);
		border-color: hsl(var(--border) / 0.5);
		color: hsl(var(--muted-foreground));
	}
	.token-chip:hover {
		background: hsl(var(--secondary));
		border-color: hsl(var(--border));
		color: hsl(var(--foreground));
	}
	.token-chip.token-current {
		background: hsl(var(--primary));
		border-color: hsl(var(--primary));
		color: hsl(var(--primary-foreground));
		box-shadow: 0 0 12px hsl(var(--primary) / 0.4);
	}
	.token-chip.token-past {
		background: hsl(var(--primary) / 0.15);
		border-color: hsl(var(--primary) / 0.4);
		color: hsl(var(--primary));
	}
	.token-chip.token-future {
		background: hsl(var(--secondary) / 0.5);
		border-color: hsl(var(--border) / 0.4);
		color: hsl(var(--muted-foreground));
	}
</style>
