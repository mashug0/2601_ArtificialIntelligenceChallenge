<script lang="ts">
	import { onMount } from 'svelte';
	import { spring } from 'svelte/motion';
	import { browser } from '$app/environment';

	// --- BDH & RoPE Engineering Constants ---
	const nodeCount = 5;
	const restRadius = 110;
	const baseFreq = 10000;
	const d = 4;

	let selectedNode = 0;
	let katexLoaded = false;

	const intensitySprings = Array.from({ length: nodeCount }, () =>
		spring(0, { stiffness: 0.1, damping: 0.3 })
	);
	let intensities = [0, 0, 0, 0, 0];
	intensitySprings.forEach((s, i) =>
		s.subscribe((v) => {
			intensities[i] = v;
		})
	);

	let particles = Array.from({ length: nodeCount }, (_, i) => {
		const angle = (i * 2 * Math.PI) / nodeCount - Math.PI / 2;
		return {
			id: i,
			x: restRadius * Math.cos(angle),
			y: restRadius * Math.sin(angle),
			phaseAngle: (i * Math.PI) / 4,
			freq: 1.0 / Math.pow(baseFreq, (2 * i) / d)
		};
	});

	let edges: { from: number; to: number }[] = [];
	for (let i = 0; i < nodeCount; i++) {
		for (let j = i + 1; j < nodeCount; j++) {
			edges.push({ from: i, to: j });
		}
	}

	function selectNode(i: number) {
		selectedNode = i;
		intensitySprings[i].set(1);
		setTimeout(() => intensitySprings[i].set(0), 400);
	}

	function getSimilarity(p1: (typeof particles)[0], p2: (typeof particles)[0]) {
		return Math.cos(p1.phaseAngle - p2.phaseAngle);
	}

	function getGlow(p1: (typeof particles)[0], p2: (typeof particles)[0]) {
		const sim = getSimilarity(p1, p2);
		return Math.max(0, (sim - 0.7) / 0.3);
	}

	// LaTeX formulas
	const formulas = {
		dimRotation: 'm \\cdot 10000^{-2i/d}',
		synapticTension: '\\sigma_{ij} = \\cos(\\theta_i - \\theta_j)',
		relativeInvariance: '\\Delta\\theta = (m - n) \\cdot \\theta',
		ropeCore:
			'\\text{RoPE}(x_m, m) = \\begin{pmatrix} x_m^{(1)} \\cos m\\theta - x_m^{(2)} \\sin m\\theta \\\\ x_m^{(1)} \\sin m\\theta + x_m^{(2)} \\cos m\\theta \\end{pmatrix}',
		dotProduct:
			'\\langle \\text{RoPE}(x_m), \\text{RoPE}(x_n) \\rangle = f(x_m, x_n, m-n)',
		freqFormula: '\\theta_i = 10000^{-2i/d}, \\quad i = 0, 1, \\ldots, d/2 - 1'
	};

	// Render LaTeX to HTML
	function renderTex(tex: string, displayMode = false): string {
		if (!katexLoaded || typeof window === 'undefined') return tex;
		try {
			return (window as any).katex.renderToString(tex, {
				throwOnError: false,
				displayMode
			});
		} catch {
			return tex;
		}
	}

	onMount(() => {
		if (browser) {
			// Load KaTeX CSS
			const link = document.createElement('link');
			link.rel = 'stylesheet';
			link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
			document.head.appendChild(link);

			// Load KaTeX JS
			const script = document.createElement('script');
			script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
			script.onload = () => {
				katexLoaded = true;
			};
			document.head.appendChild(script);
		}
	});
</script>

<svelte:head>
	<title>Rotary Positional Embeddings (RoPE) - BDH Visualization</title>
</svelte:head>

<div class="min-h-screen overflow-auto">
	<div class="max-w-7xl mx-auto px-6 py-8 lg:px-10 lg:py-12">
		<!-- Header -->
		<div class="mb-10 lg:mb-12">
			<a href="/" class="text-xs text-primary hover:text-primary/80 mb-2 inline-block font-mono tracking-[0.18em] uppercase">
				← Back to Visualization
			</a>
			<h1 class="text-2xl lg:text-3xl font-serif text-foreground mb-2">Rotary Positional Embeddings (RoPE)</h1>
			<p class="text-muted-foreground text-base max-w-2xl">
				Visualize how rotary positional embeddings encode token distance through phase rotations.
				Each node is a dimension — click to select, drag the slider to rotate its phase angle.
			</p>
		</div>

		<!-- Main: Two columns — Left: Phase Dial | Right: Synaptic Activation -->
		<div class="grid lg:grid-cols-2 gap-10 lg:gap-14 items-stretch">
			<!-- Left: Phase Dial (centred heading, dial, description) -->
			<div class="flex flex-col items-center">
				<div class="flex-1 min-h-0 w-full flex items-center justify-center">
					<div class="aspect-square w-full max-h-full flex items-center justify-center"
						style="background:
							radial-gradient(circle at top, hsl(var(--primary) / 0.06), transparent 60%),
							radial-gradient(circle at bottom, hsl(var(--accent) / 0.06), transparent 60%);"
					>
					<svg viewBox="-160 -160 320 320" class="w-full h-full">
				<!-- Reference circle -->
				<circle cx="0" cy="0" r={restRadius} fill="none" stroke="hsl(var(--muted-foreground))" stroke-width="1" stroke-dasharray="4 4" />

				{#each edges as edge}
					{@const p1 = particles[edge.from]}
					{@const p2 = particles[edge.to]}
					{@const glow = getGlow(p1, p2)}
					<line
								x1={p1.x}
								y1={p1.y}
								x2={p2.x}
								y2={p2.y}
								stroke={glow > 0.4 ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'}
								stroke-width={glow > 0.4 ? 2.5 + glow * 5 : 1.2}
								opacity={glow > 0.4 ? 0.6 : 0.4}
							/>
						{/each}

						{#each particles as p, i}
							<!-- svelte-ignore a11y-click-events-have-key-events -->
							<g on:click={() => selectNode(i)} class="cursor-pointer" role="button" tabindex="0">
								<!-- Pulse ring on click -->
								<circle
									cx={p.x}
									cy={p.y}
									r={20 + intensities[i] * 40}
									fill="hsl(var(--primary))"
									opacity={0.12 * intensities[i]}
								/>
								<!-- Node body -->
								<circle
									cx={p.x}
									cy={p.y}
									r="18"
									fill="hsl(var(--background))"
									stroke={selectedNode === i ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'}
									stroke-width={selectedNode === i ? 3 : 2}
								/>
								<!-- Phase needle -->
								<line
									x1={p.x}
									y1={p.y}
									x2={p.x + Math.cos(p.phaseAngle) * 13}
									y2={p.y + Math.sin(p.phaseAngle) * 13}
									stroke={selectedNode === i ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'}
									stroke-width="3"
									stroke-linecap="round"
								/>
								<!-- Label -->
								<text
									x={p.x}
									y={p.y + 36}
									text-anchor="middle"
									font-size="12"
									font-family="Georgia, serif"
									font-weight="600"
									fill={selectedNode === i ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'}
								>
									x{i}
								</text>
							</g>
						{/each}
					</svg>
					</div>
				</div>
				<p class="text-sm text-muted-foreground mt-6 text-center max-w-md">
					Each node represents a dimension. The needle shows its current phase angle θ.
					Connected edges glow when two nodes are in resonance (high cosine similarity).
				</p>
			</div>

			<!-- Right: Synaptic Activation Bar Chart + Slider -->
			<div class="space-y-8">
				<h3 class="text-xs font-mono text-primary uppercase tracking-widest mb-2">Synaptic Activation Potential (σ)</h3>
				<p class="text-sm text-muted-foreground mb-6">
					Cosine similarity between selected node x<sub>{selectedNode}</sub> and every other node.
					Taller bars indicate stronger resonance — the network sees those dimensions as "close."
				</p>
				<div class="flex items-end justify-around h-48 lg:h-56 gap-4 pb-2">
					{#each particles as p, i}
						{@const sim = (getSimilarity(particles[selectedNode], p) + 1) / 2}
						<div class="flex-1 rounded-t transition-all duration-300 relative min-h-[4px]"
							style="height: {Math.max(sim * 100, 4)}%; background: linear-gradient(to top, hsl(var(--primary) / 0.9), hsl(var(--primary) / 0.5));"
						>
							<span class="absolute -top-6 left-1/2 -translate-x-1/2 text-sm font-mono text-foreground font-medium whitespace-nowrap">{sim.toFixed(2)}</span>
							<span class="absolute -bottom-6 left-1/2 -translate-x-1/2 text-sm font-serif text-muted-foreground">x{i}</span>
						</div>
					{/each}
				</div>
				<p class="text-sm text-muted-foreground mt-4 text-center italic">
					Resonance spectrum for node <strong>x{selectedNode}</strong>. High bars indicate maximum synaptic
					tension (σ ≈ 1.0).
				</p>

				<!-- Phase Slider (below bar chart) -->
				<div>
					<h3 class="text-xs font-mono text-primary uppercase tracking-widest mb-1">Active Tuning</h3>
					<p class="text-xl font-serif text-foreground mb-3">Node x<sub>{selectedNode}</sub></p>
					<div class="flex justify-between text-xs font-mono text-muted-foreground uppercase mb-1">
						<span>Phase Alignment (θ)</span>
						<span class="text-primary">{(particles[selectedNode].phaseAngle * (180 / Math.PI)).toFixed(1)}°</span>
					</div>
					<input
						type="range"
						min={-Math.PI}
						max={Math.PI}
						step="0.01"
						bind:value={particles[selectedNode].phaseAngle}
						class="rope-range w-full h-2 bg-muted rounded-full appearance-none cursor-ew-resize"
					/>
					<p class="text-xs text-muted-foreground mt-2">
						Drag to rotate the phase needle. Watch edges and bars react in real time.
					</p>
				</div>
			</div>
		</div>

		<!-- Deep Dive — single column, generous spacing -->
	<div class="space-y-24 lg:space-y-32 pt-12 border-t border-border/20">

		<!-- Why RoPE? + Mathematics of RoPE — side by side, aligned -->
		<div class="grid lg:grid-cols-2 gap-10 lg:gap-14 items-start">
			<!-- Why RoPE? -->
			<section class="min-w-0">
				<h2 class="font-serif text-2xl lg:text-3xl text-foreground mb-6">Why RoPE?</h2>
				<p class="text-muted-foreground leading-relaxed mb-6">
					Standard neural networks often struggle with <strong>absolute position</strong> — they know
					<em>where</em> a word is in a sentence, but they don't naturally understand the
					<strong>distance</strong> between words. Traditional position encodings (sinusoidal or
					learned) bake absolute indices into the representation, which means the model must
					re-learn positional relationships for every sequence length.
				</p>
				<p class="text-muted-foreground leading-relaxed mb-6">
					RoPE solves this elegantly by <strong>rotating</strong> the phase dials of each embedding
					dimension, making the dot product between any two tokens a function of their
					<em>relative distance</em> alone. The result: a model that generalizes to unseen
					sequence lengths and captures both local syntax and long-range dependencies.
				</p>
				<p class="text-muted-foreground leading-relaxed mb-6">
					Beyond just fixing positions, RoPE enables <strong>context window expansion</strong>. By slightly
					"squeezing" the rotation angles—a technique called <strong>Position Interpolation</strong>—developers
					can stretch a model trained on 4,000 tokens to handle 128,000. It essentially slows down the
					"positional clock," allowing the network to navigate massive datasets without losing its
					spatial bearings.
				</p>
				<h3 class="font-serif text-base text-foreground mb-3">Example: Semantic Distance</h3>
				<p class="text-muted-foreground leading-relaxed mb-4">
					Consider the sentence: <em>"The cat sat on the mat."</em> The distance between
					<strong>'cat'</strong> and <strong>'mat'</strong> is always 4 tokens, regardless of
					where the sentence starts in a longer document.
				</p>
				<p class="text-muted-foreground leading-relaxed mb-6">
					In BDH, this distance is encoded as the <strong>phase difference (Δθ)</strong> between
					the corresponding nodes. Two tokens that are 4 positions apart will always have the same
					Δθ — the model doesn't need to memorize "position 2" and "position 6" separately.
				</p>
				{#if katexLoaded}
					<div class="py-4">
						{@html renderTex(formulas.dotProduct, true)}
					</div>
				{/if}
			</section>

			<!-- Right column: Mathematics of RoPE + Key Insights stacked -->
			<div class="min-w-0 space-y-16 lg:space-y-20">
				<!-- The Mathematics of RoPE -->
				<section>
					<h2 class="font-serif text-2xl lg:text-3xl text-foreground mb-6">The Mathematics of RoPE</h2>
					<div class="space-y-6 text-muted-foreground">
						<div>
							<p class="text-sm font-serif text-primary font-medium mb-1">1. Dimension Rotation (θ)</p>
							{#if katexLoaded}
								<div class="my-3">{@html renderTex(formulas.dimRotation, true)}</div>
							{/if}
							<p class="leading-relaxed text-sm">
								Each dimension pair rotates at a frequency determined by its index. Lower dimensions rotate
								slowly (global context), higher ones rapidly (local syntax).
							</p>
						</div>
						<div>
							<p class="text-sm font-serif text-primary font-medium mb-1">2. Synaptic Tension (σ)</p>
							{#if katexLoaded}
								<div class="my-3">{@html renderTex(formulas.synapticTension, true)}</div>
							{/if}
							<p class="leading-relaxed text-sm">
								The dot product between two rotated vectors reduces to a function of their phase
								difference — this is the core mechanism for relative position encoding.
							</p>
						</div>
						<div>
							<p class="text-sm font-serif text-primary font-medium mb-1">3. Relative Invariance</p>
							{#if katexLoaded}
								<div class="my-3">{@html renderTex(formulas.relativeInvariance, true)}</div>
							{/if}
							<p class="leading-relaxed text-sm">
								The phase difference Δθ depends only on the distance (m − n) between tokens, not
								their absolute positions. This is the key property that makes RoPE translation-invariant.
							</p>
						</div>
					</div>
				</section>

				<!-- Key Insights — right below Mathematics of RoPE -->
				<section>
					<h2 class="font-serif text-2xl lg:text-3xl text-foreground mb-6">Key Insights</h2>
					<ul class="space-y-4 text-muted-foreground">
						<li class="flex gap-2">
							<span class="text-primary font-bold">✓</span>
							<span><strong class="text-foreground">RoPE Persistence:</strong> Synaptic tension depends on token distance, not absolute position — the model sees relative structure.</span>
						</li>
						<li class="flex gap-2">
							<span class="text-primary font-bold">✓</span>
							<span><strong class="text-foreground">Multi-Scale Context:</strong> Low-frequency dimensions capture global themes; high-frequency dimensions capture local grammar and syntax.</span>
						</li>
						<li class="flex gap-2">
							<span class="text-primary font-bold">✓</span>
							<span><strong class="text-foreground">BDH Integration:</strong> In our k-WTA architecture, RoPE ensures that sparse neuron activations encode positional relationships without wasting capacity on absolute indices.</span>
						</li>
					</ul>
				</section>
			</div>
		</div>

		<!-- Full RoPE Transform + Frequency Spectrum — two columns -->
		<div class="grid lg:grid-cols-2 gap-10 lg:gap-14 items-start">
			<!-- Full RoPE Transform -->
			<section class="min-w-0">
				<h2 class="font-serif text-2xl lg:text-3xl text-foreground mb-6">The Full RoPE Transform</h2>
				<p class="text-muted-foreground leading-relaxed mb-6">
					For each position <em>m</em> in the sequence, RoPE applies a 2D rotation to every
					consecutive pair of embedding dimensions. The rotation angle is the product of the
					position index and the dimension-specific frequency:
				</p>
				{#if katexLoaded}
					<div class="py-6 mb-8">{@html renderTex(formulas.ropeCore, true)}</div>
				{/if}
				<div class="space-y-6 text-sm">
					<div>
						<h4 class="font-serif text-foreground font-medium mb-2">No Learnable Parameters</h4>
						<p class="text-muted-foreground leading-relaxed">
							RoPE is purely analytical — it requires zero additional parameters beyond the
							embedding itself. This makes it lightweight and avoids overfitting on position.
						</p>
					</div>
					<div>
						<h4 class="font-serif text-foreground font-medium mb-2">Length Generalization</h4>
						<p class="text-muted-foreground leading-relaxed">
							Because the encoding is a continuous function of position, RoPE naturally extrapolates to
							sequence lengths longer than those seen during training.
						</p>
					</div>
					<div>
						<h4 class="font-serif text-foreground font-medium mb-2">Decaying Attention</h4>
						<p class="text-muted-foreground leading-relaxed">
							The dot product between rotated embeddings naturally decays with distance for
							high-frequency dimensions, implementing an implicit attention bias toward nearby tokens.
						</p>
					</div>
				</div>
			</section>

			<!-- Frequency Spectrum -->
			<section class="min-w-0">
				<h2 class="font-serif text-2xl lg:text-3xl text-foreground mb-6">The Frequency Spectrum</h2>
				<p class="text-muted-foreground leading-relaxed mb-6">
					RoPE assigns a unique rotation frequency to each dimension pair. Lower dimensions rotate
					slowly and respond to long-range dependencies, while higher dimensions rotate rapidly and
					capture fine-grained local structure. In our {nodeCount}-node system, this creates a
					variable "stiffness" profile:
				</p>
				{#if katexLoaded}
					<div class="py-4 mb-6">{@html renderTex(formulas.freqFormula, true)}</div>
				{/if}
				<div class="space-y-8">
					<div>
						<h4 class="font-serif text-base text-foreground mb-2">
							Low Frequency (x<sub>0</sub>, x<sub>1</sub>)
						</h4>
						<p class="text-muted-foreground text-sm leading-relaxed mb-3">
							"Loose" connectors that rotate slowly. These dimensions capture <strong>long-range
							semantic dependencies</strong> — they respond similarly to tokens that are far apart,
							making them ideal for understanding document-level themes, paragraph coherence, and
							distant co-references.
						</p>
						<div class="h-1.5 w-full bg-gradient-to-r from-neon-cyan to-primary rounded-full"></div>
						<span class="text-xs font-mono text-muted-foreground mt-1 block">Slow rotation</span>
					</div>
					<div>
						<h4 class="font-serif text-base text-foreground mb-2">
							High Frequency (x<sub>3</sub>, x<sub>4</sub>)
						</h4>
						<p class="text-muted-foreground text-sm leading-relaxed mb-3">
							"Stiff" connectors that rotate rapidly. These dimensions react sharply to
							<strong>immediate word order and local grammar</strong> — adjacent tokens produce
							large phase differences, making them sensitive to syntax, morphology, and local
							word patterns.
						</p>
						<div class="h-1.5 w-full bg-gradient-to-r from-accent to-neon-amber rounded-full"></div>
						<span class="text-xs font-mono text-muted-foreground mt-1 block">Fast rotation</span>
					</div>
				</div>
			</section>
		</div>
	</div>
	</div>
</div>

<style>
	.rope-range::-webkit-slider-thumb {
		appearance: none;
		width: 20px;
		height: 20px;
		background: hsl(var(--primary));
		border-radius: 50%;
		border: 3px solid hsl(var(--surface));
		cursor: ew-resize;
	}
	.rope-range::-moz-range-thumb {
		width: 20px;
		height: 20px;
		background: hsl(var(--primary));
		border-radius: 50%;
		border: 3px solid hsl(var(--surface));
		cursor: ew-resize;
	}
</style>