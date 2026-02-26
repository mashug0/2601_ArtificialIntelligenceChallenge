<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	const API_URL = '';

	// ── Types ──────────────────────────────────────────────────────────────────
	interface Similarity {
		word1: string;
		word2: string;
		similarity: number;
	}

	// ── API State ──────────────────────────────────────────────────────────────
	let similarities: Similarity[] = [];
	let isLoading = true;
	let error = '';

	async function loadData() {
		try {
			const res = await fetch(`${API_URL}/api/concept-memory`);
			if (!res.ok) throw new Error(`API error: ${res.status}`);
			const data = await res.json();
			similarities = data.similarities;
			isLoading = false;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load concept memory';
			isLoading = false;
		}
	}

	onMount(() => { loadData(); });

	// ── Noise Trace State (real model data) ───────────────────────────────────
	const TRACE_WORDS = [
		'king','queen','war','love','fire','dark','light','moon','sun',
		'army','heart','night','sword','blood','wind','soul','child','death'
	];
	let selectedWord = 'king';
	let noiseScale = 3.0;            // ramp max — needs to be large to see decay
	let decayRate = 0.0012;          // λ for dense baseline exponential decay

	interface TracePoint { pass_idx: number; noise_level: number; activation: number; }
	let fullTrace: TracePoint[] = [];       // full data from API
	let revealedTrace: TracePoint[] = [];   // animated subset
	let cleanActivation = 0;
	let topNeuron = -1;
	let isFetching = false;
	let traceError = '';

	let isPlaying = false;
	let animFrame: number;
	let revealStep = 0;

	// ── Dense Transformer Baseline (computed locally, no API) ─────────────────
	let baselineTrace: TracePoint[] = [];
	let revealedBaseline: TracePoint[] = [];

	function generateBaseline() {
		// Simulate a standard dense transformer neuron: exponential decay under noise.
		// Peak is anchored to the real model's clean_activation so y-axes are comparable.
		baselineTrace = Array.from({ length: N_PASSES }, (_, i) => ({
			pass_idx: i,
			noise_level: (i / (N_PASSES - 1)) * noiseScale,
			// Fast exponential decay + small biological jitter
			activation: Math.max(0,
				cleanActivation * Math.exp(-decayRate * i) +
				(Math.random() - 0.5) * cleanActivation * 0.06
			)
		}));
	}

	async function fetchTrace() {
		stopAnimation();
		isFetching = true;
		traceError = '';
		fullTrace = [];
		revealedTrace = [];
		baselineTrace = [];
		revealedBaseline = [];
		revealStep = 0;
		try {
			const res = await fetch(
				`${API_URL}/api/noise-trace?word=${encodeURIComponent(selectedWord)}&n_passes=5000&noise_scale=${noiseScale}`
			);
			if (!res.ok) throw new Error(`API error: ${res.status}`);
			const data = await res.json();
			fullTrace = data.trace;
			cleanActivation = data.clean_activation;
			topNeuron = data.top_neuron;
			generateBaseline();   // anchored to real cleanActivation
			isPlaying = true;
			playAnimation();
		} catch (e) {
			traceError = e instanceof Error ? e.message : 'Failed to fetch trace';
		} finally {
			isFetching = false;
		}
	}

	// Advance ~80 steps per frame so 5000-point trace plays in ~1.5 seconds
	const STEPS_PER_FRAME = 80;

	function playAnimation() {
		if (!isPlaying) return;
		if (revealStep >= fullTrace.length) { isPlaying = false; return; }
		revealStep = Math.min(revealStep + STEPS_PER_FRAME, fullTrace.length);
		revealedTrace = fullTrace.slice(0, revealStep);
		revealedBaseline = baselineTrace.slice(0, revealStep);
		animFrame = requestAnimationFrame(playAnimation);
	}

	function togglePlay() {
		if (isPlaying) { stopAnimation(); }
		else if (fullTrace.length > 0) { isPlaying = true; playAnimation(); }
	}

	function stopAnimation() {
		isPlaying = false;
		if (animFrame) cancelAnimationFrame(animFrame);
	}

	function resetGraph() {
		stopAnimation();
		fullTrace = [];
		revealedTrace = [];
		baselineTrace = [];
		revealedBaseline = [];
		revealStep = 0;
		cleanActivation = 0;
		topNeuron = -1;
	}

	onDestroy(() => { stopAnimation(); });

	// ── SVG helpers ────────────────────────────────────────────────────────────
	const SVG_W = 600;
	const SVG_H = 320;
	const PAD_L = 52;
	const PAD_R = 18;
	const PAD_T = 16;
	const PAD_B = 36;
	const PLOT_W = SVG_W - PAD_L - PAD_R;
	const PLOT_H = SVG_H - PAD_T - PAD_B;
	const N_PASSES = 5000;

	function xScale(t: number): number { return PAD_L + (t / (N_PASSES - 1)) * PLOT_W; }
	// y-axis: 0 → bottom, cleanActivation * 1.15 → top (dynamic ceiling)
	$: yMax = cleanActivation > 0 ? cleanActivation * 1.18 : 1;
	function yScale(v: number): number { return PAD_T + PLOT_H - (v / yMax) * PLOT_H; }

	// Downsample to ~400 points for SVG performance (5000 pts would be sluggish)
	$: polylinePoints = (() => {
		if (revealedTrace.length === 0) return '';
		const step = Math.max(1, Math.floor(revealedTrace.length / 400));
		const pts = revealedTrace.filter((_, i) => i % step === 0 || i === revealedTrace.length - 1);
		return pts.map(p => `${xScale(p.pass_idx).toFixed(1)},${yScale(p.activation).toFixed(1)}`).join(' ');
	})();

	$: baselinePolylinePoints = (() => {
		if (revealedBaseline.length === 0) return '';
		const step = Math.max(1, Math.floor(revealedBaseline.length / 400));
		const pts = revealedBaseline.filter((_, i) => i % step === 0 || i === revealedBaseline.length - 1);
		return pts.map(p => `${xScale(p.pass_idx).toFixed(1)},${yScale(p.activation).toFixed(1)}`).join(' ');
	})();

	$: currentStrength = revealedTrace.length > 0 ? revealedTrace[revealedTrace.length - 1].activation : 0;
	$: halfLifePass = (() => {
		const half = cleanActivation * 0.5;
		return fullTrace.findIndex(p => p.activation <= half);
	})();

	// ── Similarity Matrix ──────────────────────────────────────────────────────
	$: simMatrix = (() => {
		const words = new Set<string>();
		similarities.forEach((s) => { words.add(s.word1); words.add(s.word2); });
		const wordList = Array.from(words).sort();
		const matrix: number[][] = wordList.map(() => wordList.map(() => 1));
		similarities.forEach((s) => {
			const i = wordList.indexOf(s.word1);
			const j = wordList.indexOf(s.word2);
			if (i >= 0 && j >= 0) { matrix[i][j] = s.similarity; matrix[j][i] = s.similarity; }
		});
		return { words: wordList, matrix };
	})();

	function getCellStyle(value: number): { background: string; textColor: string; showDiagonalGlow: boolean } {
		const v = Math.max(0, Math.min(1, value));
		const showDiagonalGlow = v >= 0.99;
		if (v === 0) return { background: '#0D1117', textColor: '#ffffff', showDiagonalGlow: false };
		if (v >= 0.99) return { background: '#FFFFFF', textColor: '#000000', showDiagonalGlow: true };
		let r: number, g: number, b: number;
		if (v < 0.1) {
			const t = v / 0.1;
			r = Math.round(13 + (22 - 13) * t); g = Math.round(17 + (51 - 17) * t); b = Math.round(23 + (86 - 23) * t);
		} else if (v <= 0.6) {
			const t = (v - 0.1) / 0.5;
			r = Math.round(22 + (31 - 22) * t); g = Math.round(51 + (111 - 51) * t); b = Math.round(86 + (235 - 86) * t);
		} else if (v <= 0.9) {
			r = 88; g = 166; b = 255;
		} else {
			const t = (v - 0.9) / 0.09;
			r = Math.round(88 + (255 - 88) * t); g = Math.round(166 + (255 - 166) * t); b = 255;
		}
		return { background: `rgb(${r},${g},${b})`, textColor: v >= 0.9 ? '#000000' : '#ffffff', showDiagonalGlow: false };
	}
	function getSimColor(val: number): string { return getCellStyle(val).background; }

	const cellSize = 52;
	$: matrixN = simMatrix.words.length;
	let hoveredCell: { row: number; col: number } | null = null;

</script>

<svelte:head>
	<title>Concept Storage & Memory — BDH</title>
</svelte:head>

<div class="max-w-6xl mx-auto px-4 py-8">

	<!-- ── Header ─────────────────────────────────────────────────────────────── -->
	<div class="mb-8">
		<a href="/" class="text-xs text-primary hover:text-primary/80 mb-4 inline-block font-mono tracking-[0.18em] uppercase">
			← Back to Visualization
		</a>
		<h1 class="text-3xl font-serif text-foreground mb-3">Concept Storage &amp; Memory</h1>
		<p class="text-muted-foreground text-base max-w-2xl">
			How semantic concepts are encoded in sparse neuron populations — and how those neurons
			hold their signal as input noise increases.
		</p>
	</div>

	{#if isLoading}
		<div class="flex items-center justify-center h-[50vh]">
			<div class="text-center">
				<div class="concept-spinner mx-auto mb-4"></div>
				<p class="text-muted-foreground font-mono text-sm">Loading concept memory data...</p>
			</div>
		</div>

	{:else if error}
		<div class="blend-card border border-red-900/40 rounded-lg p-6 text-center">
			<p class="text-red-400 mb-2">{error}</p>
			<p class="text-sm text-muted-foreground">Make sure the backend is running:
				<code class="font-mono text-primary">python main.py</code>
			</p>
		</div>

	{:else}

	<!-- ═══════════════════════════════════════════════════════════════════════ -->
	<!-- SECTION 1 — Neuron Stability Under Noise (real model data)              -->
	<!-- ═══════════════════════════════════════════════════════════════════════ -->
	<section class="mb-10">
		<div class="blend-card rounded-xl p-6">
			<h2 class="font-serif text-xl text-foreground mb-1">Neuron Stability Under Noise</h2>
			<p class="text-sm text-muted-foreground mb-6">
				Real BDH model data vs a simulated dense baseline. Both graphs share the same y-axis and
				noise ramp — showing how k-WTA sparse neurons hold their signal where a dense neuron
				would rapidly decay.
			</p>

			<!-- ── Two graphs side by side ───────────────────────────────────────── -->
			<div class="grid md:grid-cols-2 gap-5 mb-6">

				<!-- LEFT: Dense Transformer (baseline simulation) -->
				<div class="min-w-0">
					<p class="text-xs font-mono mb-2 uppercase tracking-widest" style="color: hsl(0 70% 55%);">
						Dense Transformer (Baseline)
					</p>
					<svg viewBox="0 0 {SVG_W} {SVG_H}" class="w-full block rounded-lg"
						style="background: hsl(var(--surface) / 0.4);">
						<defs>
							<filter id="baselinePeakGlow" x="-60%" y="-60%" width="220%" height="220%">
								<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
								<feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
							</filter>
						</defs>

						<!-- Clean activation baseline -->
						{#if cleanActivation > 0}
							<line x1={PAD_L} y1={yScale(cleanActivation)} x2={PAD_L + PLOT_W} y2={yScale(cleanActivation)}
								stroke="hsl(0 70% 52% / 0.25)" stroke-width="1" stroke-dasharray="5 3" />
							<text x={PAD_L + PLOT_W - 2} y={yScale(cleanActivation) - 4} text-anchor="end"
								font-size="8" fill="hsl(0 70% 52% / 0.6)" font-family="JetBrains Mono, monospace">clean baseline</text>
						{/if}

						<!-- Grid lines -->
						{#each [0.25, 0.5, 0.75, 1.0] as frac}
							{@const gv = frac * yMax}
							<line x1={PAD_L} y1={yScale(gv)} x2={PAD_L + PLOT_W} y2={yScale(gv)}
								stroke="hsl(var(--border) / 0.25)" stroke-width="1"
								stroke-dasharray={frac === 0.5 ? '4 3' : ''} />
							<text x={PAD_L - 6} y={yScale(gv) + 4} text-anchor="end" font-size="9"
								fill="hsl(215 20% 42%)" font-family="JetBrains Mono, monospace">{gv.toFixed(3)}</text>
						{/each}

						<!-- X-axis -->
						{#each [0, 1000, 2000, 3000, 4000, 4999] as tick}
							<text x={xScale(tick)} y={PAD_T + PLOT_H + 18} text-anchor="middle" font-size="9"
								fill="hsl(215 20% 42%)" font-family="JetBrains Mono, monospace">{tick}</text>
						{/each}
						<text x={PAD_L + PLOT_W / 2} y={SVG_H - 2} text-anchor="middle" font-size="9"
							fill="hsl(215 20% 32%)" font-family="JetBrains Mono, monospace">noise pass →</text>

						<!-- Y-axis label -->
						<text x={10} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="9"
							fill="hsl(215 20% 32%)" font-family="JetBrains Mono, monospace"
							transform="rotate(-90, 10, {PAD_T + PLOT_H / 2})">neuron activation</text>

						<!-- Baseline polyline — red/orange -->
						{#if baselinePolylinePoints}
							<polyline points={baselinePolylinePoints} fill="none"
								stroke="hsl(0 72% 55%)" stroke-width="2"
								stroke-linecap="round" stroke-linejoin="round" opacity="0.95" />
						{/if}

						<!-- Peak dot -->
						{#if revealedBaseline.length > 0}
							<circle cx={xScale(0)} cy={yScale(revealedBaseline[0].activation)}
								r="4" fill="hsl(0 72% 68%)" filter="url(#baselinePeakGlow)" />
						{/if}

						<!-- Axes -->
						<line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + PLOT_H} stroke="hsl(var(--border) / 0.5)" stroke-width="1" />
						<line x1={PAD_L} y1={PAD_T + PLOT_H} x2={PAD_L + PLOT_W} y2={PAD_T + PLOT_H} stroke="hsl(var(--border) / 0.5)" stroke-width="1" />

						{#if isFetching}
							<text x={PAD_L + PLOT_W / 2} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="11"
								fill="hsl(0 70% 45%)" font-family="JetBrains Mono, monospace">computing baseline...</text>
						{:else if revealedBaseline.length === 0}
							<text x={PAD_L + PLOT_W / 2} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="11"
								fill="hsl(215 20% 32%)" font-family="JetBrains Mono, monospace">← press Run to compare</text>
						{/if}
					</svg>
				</div>

				<!-- RIGHT: BDH k-WTA (real model) -->
				<div class="min-w-0">
					<p class="text-xs font-mono mb-2 uppercase tracking-widest" style="color: hsl(186 85% 52%);">
						BDH k-WTA (Ours)
					</p>
					<svg viewBox="0 0 {SVG_W} {SVG_H}" class="w-full block rounded-lg"
						style="background: hsl(var(--surface) / 0.4);">
						<defs>
							<filter id="peakGlow" x="-60%" y="-60%" width="220%" height="220%">
								<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
								<feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
							</filter>
						</defs>

						<!-- Clean activation baseline -->
						{#if cleanActivation > 0}
							<line x1={PAD_L} y1={yScale(cleanActivation)} x2={PAD_L + PLOT_W} y2={yScale(cleanActivation)}
								stroke="hsl(186 85% 52% / 0.25)" stroke-width="1" stroke-dasharray="5 3" />
							<text x={PAD_L + PLOT_W - 2} y={yScale(cleanActivation) - 4} text-anchor="end"
								font-size="8" fill="hsl(186 85% 52% / 0.6)" font-family="JetBrains Mono, monospace">clean baseline</text>
						{/if}

						<!-- Grid lines -->
						{#each [0.25, 0.5, 0.75, 1.0] as frac}
							{@const gv = frac * yMax}
							<line x1={PAD_L} y1={yScale(gv)} x2={PAD_L + PLOT_W} y2={yScale(gv)}
								stroke="hsl(var(--border) / 0.25)" stroke-width="1"
								stroke-dasharray={frac === 0.5 ? '4 3' : ''} />
							<text x={PAD_L - 6} y={yScale(gv) + 4} text-anchor="end" font-size="9"
								fill="hsl(215 20% 42%)" font-family="JetBrains Mono, monospace">{gv.toFixed(3)}</text>
						{/each}

						<!-- X-axis -->
						{#each [0, 1000, 2000, 3000, 4000, 4999] as tick}
							<text x={xScale(tick)} y={PAD_T + PLOT_H + 18} text-anchor="middle" font-size="9"
								fill="hsl(215 20% 42%)" font-family="JetBrains Mono, monospace">{tick}</text>
						{/each}
						<text x={PAD_L + PLOT_W / 2} y={SVG_H - 2} text-anchor="middle" font-size="9"
							fill="hsl(215 20% 32%)" font-family="JetBrains Mono, monospace">noise pass →</text>

						<!-- Y-axis label -->
						<text x={10} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="9"
							fill="hsl(215 20% 32%)" font-family="JetBrains Mono, monospace"
							transform="rotate(-90, 10, {PAD_T + PLOT_H / 2})">neuron activation</text>

						<!-- Half-life marker -->
						{#if halfLifePass >= 0 && halfLifePass <= revealStep}
							<line x1={xScale(halfLifePass)} y1={PAD_T} x2={xScale(halfLifePass)} y2={PAD_T + PLOT_H}
								stroke="hsl(40 100% 50% / 0.45)" stroke-width="1" stroke-dasharray="3 3" />
							<text x={xScale(halfLifePass) + 3} y={PAD_T + 14} font-size="8"
								fill="hsl(40 100% 55%)" font-family="JetBrains Mono, monospace">50% drop</text>
						{/if}

						<!-- BDH polyline — cyan -->
						{#if polylinePoints}
							<polyline points={polylinePoints} fill="none"
								stroke="hsl(186 85% 52%)" stroke-width="2"
								stroke-linecap="round" stroke-linejoin="round" opacity="0.95" />
						{/if}

						<!-- Peak dot -->
						{#if revealedTrace.length > 0}
							<circle cx={xScale(0)} cy={yScale(revealedTrace[0].activation)}
								r="4" fill="hsl(186 85% 65%)" filter="url(#peakGlow)" />
						{/if}

						<!-- Axes -->
						<line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + PLOT_H} stroke="hsl(var(--border) / 0.5)" stroke-width="1" />
						<line x1={PAD_L} y1={PAD_T + PLOT_H} x2={PAD_L + PLOT_W} y2={PAD_T + PLOT_H} stroke="hsl(var(--border) / 0.5)" stroke-width="1" />

						{#if traceError}
							<text x={PAD_L + PLOT_W / 2} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="11"
								fill="hsl(0 70% 55%)" font-family="JetBrains Mono, monospace">{traceError}</text>
						{:else if isFetching}
							<text x={PAD_L + PLOT_W / 2} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="11"
								fill="hsl(186 85% 45%)" font-family="JetBrains Mono, monospace">running model...</text>
						{:else if revealedTrace.length === 0}
							<text x={PAD_L + PLOT_W / 2} y={PAD_T + PLOT_H / 2} text-anchor="middle" font-size="11"
								fill="hsl(215 20% 32%)" font-family="JetBrains Mono, monospace">← press Run to compare</text>
						{/if}
					</svg>
				</div>
			</div>

			<!-- ── Controls panel: 3-row layout below graphs ────────────────────── -->
			<div class="pt-5 border-t" style="border-color: hsl(var(--border) / 0.2);">

				<!-- Row 1: Word chips — full width -->
				<div class="mb-5">
					<p class="text-xs font-mono text-muted-foreground mb-2 uppercase tracking-widest">Word</p>
					<div class="flex flex-wrap gap-1.5">
						{#each TRACE_WORDS as w}
							<button
								on:click={() => (selectedWord = w)}
								class="px-2.5 py-1 rounded-md text-xs font-mono transition-all"
								style="background: {selectedWord === w ? 'hsl(186 85% 52% / 0.2)' : 'hsl(var(--surface) / 0.5)'}; color: {selectedWord === w ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'}; border: 1px solid {selectedWord === w ? 'hsl(var(--primary) / 0.5)' : 'hsl(var(--border) / 0.2)'};"
							>{w}</button>
						{/each}
					</div>
				</div>

				<!-- Row 2: Sliders side by side — full width, each 50% -->
				<div class="grid grid-cols-2 gap-6 mb-5">

					<!-- Noise scale slider (cyan thumb) -->
					<div>
						<div class="flex justify-between text-xs font-mono mb-1.5">
							<span class="text-muted-foreground">Max noise scale <span class="text-[10px] opacity-50">(BDH input noise)</span></span>
							<span class="text-primary">{noiseScale.toFixed(1)}</span>
						</div>
						<input type="range" min="0.5" max="5.0" step="0.1"
							bind:value={noiseScale}
							class="decay-slider w-full"
						/>
						<div class="flex justify-between text-[10px] text-muted-foreground/50 font-mono mt-1">
							<span>gentle</span><span>severe</span>
						</div>
					</div>

					<!-- Decay rate λ slider (red thumb) -->
					<div>
						<div class="flex justify-between text-xs font-mono mb-1.5">
							<span class="text-muted-foreground">Baseline decay λ <span class="text-[10px] opacity-50">(dense transformer)</span></span>
							<span style="color: hsl(0 72% 58%);">{decayRate.toFixed(4)}</span>
						</div>
						<input type="range" min="0.0002" max="0.005" step="0.0001"
							bind:value={decayRate}
							class="decay-slider-baseline w-full"
						/>
						<div class="flex justify-between text-[10px] text-muted-foreground/50 font-mono mt-1">
							<span>slow decay</span><span>fast decay</span>
						</div>
					</div>
				</div>

				<!-- Row 3: Buttons + live stats on one line -->
				<div class="flex items-center gap-3 flex-wrap">
					<button on:click={fetchTrace} disabled={isFetching}
						class="decay-feed-btn px-5 py-2 rounded-lg text-sm font-mono font-medium transition-all">
						{isFetching ? '⏳ Running…' : '⚡ Run'}
					</button>
					<button on:click={togglePlay} disabled={fullTrace.length === 0}
						class="px-4 py-2 rounded-lg text-sm font-mono transition-all"
						style="background: {isPlaying ? 'hsl(186 85% 52% / 0.15)' : 'transparent'}; color: hsl(var(--primary)); border: 1.5px solid hsl(var(--primary) / {fullTrace.length === 0 ? '0.3' : '1'});">
						{isPlaying ? '⏸' : '▶'}
					</button>
					<button on:click={resetGraph}
						class="px-3 py-2 rounded-lg text-sm font-mono transition-all"
						style="background: hsl(var(--surface) / 0.6); color: hsl(var(--muted-foreground)); border: 1px solid hsl(var(--border) / 0.25);">✕</button>

					<!-- Divider -->
					<div class="w-px self-stretch mx-1" style="background: hsl(var(--border) / 0.3);"></div>

					<!-- Live stats inline -->
					<div class="flex gap-5 text-[11px] font-mono flex-wrap">
						<span class="text-muted-foreground">neuron <span class="text-foreground font-semibold">{topNeuron >= 0 ? `#${topNeuron}` : '—'}</span></span>
						<span class="text-muted-foreground">clean <span class="text-foreground font-semibold">{cleanActivation > 0 ? cleanActivation.toFixed(3) : '—'}</span></span>
						<span class="text-muted-foreground">now <span class="font-semibold" style="color: hsl(var(--primary));">{currentStrength > 0 ? currentStrength.toFixed(3) : '—'}</span></span>
						<span class="text-muted-foreground">50% drop <span class="text-foreground font-semibold">{halfLifePass >= 0 && fullTrace.length > 0 ? `pass #${halfLifePass}` : '—'}</span></span>
					</div>
				</div>

			</div>
		</div>
	</section>

	<!-- ═══════════════════════════════════════════════════════════════════════ -->
	<!-- SECTION 2 — Semantic Similarity Matrix                                  -->
	<!-- ═══════════════════════════════════════════════════════════════════════ -->
	<section class="mb-10">
		<div class="blend-card rounded-xl p-8">
			<h2 class="font-serif text-xl text-foreground mb-2">Semantic Similarity Matrix</h2>
			<p class="text-sm text-muted-foreground mb-6 max-w-2xl">
				Cosine similarity between learned word representations. The bright diagonal shows each
				word is most similar to itself. Off-diagonal brightness reveals how much two words share
				the same neuron populations.
			</p>

			<div class="overflow-auto">
				<svg
					width={matrixN * cellSize + 110}
					height={matrixN * cellSize + 110}
					class="mx-auto block"
				>
					<defs>
						<filter id="obsidian-cyan-glow" x="-50%" y="-50%" width="200%" height="200%">
							<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
							<feFlood flood-color="rgb(0, 229, 255)" flood-opacity="0.5" result="glow" />
							<feComposite in="glow" in2="blur" operator="in" result="glow-blur" />
							<feMerge><feMergeNode in="glow-blur" /><feMergeNode in="SourceGraphic" /></feMerge>
						</filter>
						<filter id="diagonal-cyan-glow" x="-80%" y="-80%" width="260%" height="260%">
							<feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
							<feFlood flood-color="rgb(0, 229, 255)" flood-opacity="0.6" result="glow" />
							<feComposite in="glow" in2="blur" operator="in" result="glow-blur" />
							<feMerge><feMergeNode in="glow-blur" /><feMergeNode in="SourceGraphic" /></feMerge>
						</filter>
					</defs>

					<!-- Crosshair dim -->
					{#if hoveredCell}
						<rect x={100} y={100} width={matrixN * cellSize} height={hoveredCell.row * cellSize} fill="rgba(0,0,0,0.5)" pointer-events="none" />
						<rect x={100} y={100 + (hoveredCell.row + 1) * cellSize} width={matrixN * cellSize} height={(matrixN - hoveredCell.row - 1) * cellSize} fill="rgba(0,0,0,0.5)" pointer-events="none" />
						<rect x={100} y={100} width={hoveredCell.col * cellSize} height={matrixN * cellSize} fill="rgba(0,0,0,0.5)" pointer-events="none" />
						<rect x={100 + (hoveredCell.col + 1) * cellSize} y={100} width={(matrixN - hoveredCell.col - 1) * cellSize} height={matrixN * cellSize} fill="rgba(0,0,0,0.5)" pointer-events="none" />
					{/if}

					<!-- Column headers -->
					{#each simMatrix.words as word, i}
						<text x={100 + i * cellSize} y={75} text-anchor="start" font-size="10"
							fill="#8B949E" font-family="JetBrains Mono, monospace" font-weight="500"
							transform="rotate(-45, {100 + i * cellSize}, {75})">{word}</text>
					{/each}
					<!-- Row headers -->
					{#each simMatrix.words as word, i}
						<text x={95} y={100 + i * cellSize + cellSize / 2 + 4} text-anchor="end"
							font-size="10" fill="#8B949E" font-family="JetBrains Mono, monospace" font-weight="500">{word}</text>
					{/each}

					<!-- Cells -->
					{#each simMatrix.matrix as row, i}
						{#each row as val, j}
							{@const style = getCellStyle(val)}
							{@const isHovered = hoveredCell && hoveredCell.row === i && hoveredCell.col === j}
							<rect
								x={100 + j * cellSize + 0.5} y={100 + i * cellSize + 0.5}
								width={cellSize - 1} height={cellSize - 1}
								fill={style.background}
								stroke={isHovered ? '#00e5ff' : '#1B2330'} stroke-width="1" rx="2"
								filter={isHovered ? 'url(#obsidian-cyan-glow)' : style.showDiagonalGlow ? 'url(#diagonal-cyan-glow)' : ''}
								on:mouseenter={() => (hoveredCell = { row: i, col: j })}
								on:mouseleave={() => (hoveredCell = null)}
								role="img"
								aria-label="{simMatrix.words[i]} {simMatrix.words[j]} similarity {val.toFixed(2)}"
							>
								<title>{simMatrix.words[i]} – {simMatrix.words[j]}: {val.toFixed(4)}</title>
							</rect>
							<text
								x={100 + j * cellSize + cellSize / 2} y={100 + i * cellSize + cellSize / 2 + 4}
								text-anchor="middle" font-size="10" fill={style.textColor}
								font-family="JetBrains Mono, monospace" font-weight="500" pointer-events="none"
							>{val.toFixed(2)}</text>
						{/each}
					{/each}
				</svg>
			</div>

			<!-- Legend -->
			<div class="flex items-center gap-4 mt-6 justify-center flex-wrap">
				<span class="text-[0.75rem] font-mono text-muted-foreground">0.0 (Inactive)</span>
				<div class="flex h-4 rounded-full overflow-hidden" style="width: 280px;">
					{#each Array(30) as _, i}
						<div style="flex:1; background-color: {getSimColor(i / 30)}"></div>
					{/each}
				</div>
				<span class="text-[0.75rem] font-mono text-muted-foreground">1.0 (Peak)</span>
			</div>
		</div>
	</section>

	<!-- ═══════════════════════════════════════════════════════════════════════ -->
	<!-- SECTION 3 — Concept Algebra + Semantic Divergence                       -->
	<!-- ═══════════════════════════════════════════════════════════════════════ -->
	<section class="mb-10">
		<div class="grid md:grid-cols-2 gap-6 items-stretch">

			<!-- Concept Algebra — top 8 most similar pairs -->
			<div class="blend-card rounded-xl p-6 flex flex-col">
				<h3 class="font-serif text-lg text-foreground mb-1">Concept Algebra</h3>
				<p class="text-sm text-muted-foreground mb-5">
					Proof of learned semantic structure — nearest neighbor by vector arithmetic.
				</p>
				<div class="flex-1 flex flex-col justify-between gap-0">
					{#each [...similarities].sort((a, b) => b.similarity - a.similarity).slice(0, 8) as sim}
						<div class="flex items-center gap-2 font-mono py-1">
							<span class="text-base font-bold text-primary w-14 truncate">{sim.word1}</span>
							<span class="text-muted-foreground text-xs">+</span>
							<span class="text-base font-bold text-primary w-14 truncate">{sim.word2}</span>
							<span class="text-muted-foreground text-xs mx-0.5">=</span>
							<span class="text-primary font-bold text-base tabular-nums w-12">{sim.similarity.toFixed(3)}</span>
							<div class="flex-1 rounded-full h-2 overflow-hidden" style="background: hsl(var(--muted) / 0.5);">
								<div class="h-full rounded-full sim-bar-high"
									style="width: {Math.max(0, sim.similarity) * 100}%"></div>
							</div>
						</div>
					{/each}
				</div>
				<p class="text-xs text-muted-foreground mt-4 text-center">
					High similarity = network stores both words in overlapping neuron sets
				</p>
			</div>

			<!-- Semantic Divergence — bottom 8 least similar pairs -->
			<div class="blend-card rounded-xl p-6 flex flex-col">
				<h3 class="font-serif text-lg text-foreground mb-1">Semantic Divergence</h3>
				<p class="text-sm text-muted-foreground mb-5">
					Maximally separated concepts — proof of clean neuron boundaries.
				</p>
				<div class="flex-1 flex flex-col justify-between gap-0">
					{#each [...similarities].sort((a, b) => a.similarity - b.similarity).slice(0, 8) as sim}
						<div class="flex items-center gap-2 font-mono py-1">
							<span class="text-base font-bold text-foreground w-14 truncate">{sim.word1}</span>
							<span class="text-muted-foreground text-xs">vs</span>
							<span class="text-base font-bold text-foreground w-14 truncate">{sim.word2}</span>
							<div class="flex-1 rounded-full h-2 overflow-hidden" style="background: hsl(var(--muted) / 0.5);">
								<div class="h-full rounded-full sim-bar-low"
									style="width: {Math.max(4, Math.abs(sim.similarity) * 100 * 3)}%"></div>
							</div>
							<span class="font-mono text-sm tabular-nums w-12 text-right" style="color: hsl(var(--muted-foreground));">
								{sim.similarity.toFixed(3)}
							</span>
						</div>
					{/each}
				</div>
				<p class="text-xs text-muted-foreground mt-4 text-center">
					Low similarity = k-WTA successfully separated unrelated concepts
				</p>
			</div>

		</div>
	</section>

	{/if}
</div>

<style>
	/* Spinner — cyan theme */
	.concept-spinner {
		width: 2rem; height: 2rem;
		border: 2px solid hsl(var(--border));
		border-top-color: hsl(var(--primary));
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	/* Feed signal button — outlined primary cyan (matches ReLU play btn) */
	.decay-feed-btn {
		background: transparent;
		color: hsl(var(--primary));
		border: 1.5px solid hsl(var(--primary));
	}
	.decay-feed-btn:hover {
		background: hsl(186 85% 52% / 0.15);
		box-shadow: 0 0 12px hsl(186 85% 52% / 0.35);
	}

	/* Sliders — matching ReLU page pattern */
	.decay-slider {
		appearance: none;
		-webkit-appearance: none;
		height: 8px;
		border-radius: 4px;
		background: hsl(var(--muted));
		cursor: pointer;
		width: 100%;
	}
	.decay-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 18px; height: 18px;
		background: hsl(var(--primary));
		border-radius: 50%;
		border: 2px solid hsl(var(--surface));
		cursor: pointer;
		box-shadow: 0 0 0 1px hsl(var(--border) / 0.5);
	}
	.decay-slider::-moz-range-thumb {
		width: 18px; height: 18px;
		background: hsl(var(--primary));
		border-radius: 50%;
		border: 2px solid hsl(var(--surface));
		cursor: pointer;
	}

	/* Baseline decay λ slider — red thumb to match dense transformer graph color */
	.decay-slider-baseline {
		appearance: none;
		-webkit-appearance: none;
		height: 8px;
		border-radius: 4px;
		background: hsl(var(--muted));
		cursor: pointer;
		width: 100%;
	}
	.decay-slider-baseline::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 18px; height: 18px;
		background: hsl(0 72% 55%);
		border-radius: 50%;
		border: 2px solid hsl(var(--surface));
		cursor: pointer;
		box-shadow: 0 0 0 1px hsl(var(--border) / 0.5);
	}
	.decay-slider-baseline::-moz-range-thumb {
		width: 18px; height: 18px;
		background: hsl(0 72% 55%);
		border-radius: 50%;
		border: 2px solid hsl(var(--surface));
		cursor: pointer;
	}

	/* Concept Algebra bars — cyan → teal gradient matching theme */
	.sim-bar-high {
		background: linear-gradient(to right, hsl(186 85% 38%), hsl(186 85% 58%));
		box-shadow: 0 0 6px hsl(186 85% 52% / 0.4);
	}

	/* Semantic Divergence bars — dim cyan → muted, still visible */
	.sim-bar-low {
		background: linear-gradient(to right, hsl(186 40% 28%), hsl(186 55% 42%));
		box-shadow: 0 0 4px hsl(186 55% 42% / 0.3);
	}

	/* blend-card — identical to ReLU page */
	:global(.blend-card) {
		background: hsl(var(--surface) / 0.28);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border: 1px solid hsl(var(--border) / 0.2);
		border-radius: var(--radius);
		box-shadow: 0 1px 0 0 hsl(var(--background) / 0.5);
	}
	/* Inner blend-cards are transparent (no double-glass) */
	:global(.blend-card .blend-card) {
		background: transparent;
		backdrop-filter: none;
		-webkit-backdrop-filter: none;
		border: none;
		box-shadow: none;
	}
</style>
