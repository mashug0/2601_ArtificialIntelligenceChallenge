<script lang="ts">
	import { onMount } from 'svelte';
	import { spring } from 'svelte/motion';

	// Animation state
	let isPlaying = false;
	let animationFrame: number;
	let currentStep = 0;
	let time = 0;

	// Network dimensions
	const D = 4; // Hidden dimension (simplified)
	const N = 8; // Neuronal dimension (simplified)

	// Threshold
	let threshold = spring(0.5, { stiffness: 0.05, damping: 0.5 });
	let thresholdValue = 0.5;

	// Neuron activations (will be computed)
	let neuronActivations: number[] = Array(N).fill(0);
	let sparseActivations: number[] = Array(N).fill(0);
	let sparsity = 0;

	// Hidden state (random initialization for demo)
	let hiddenState: number[] = Array(D)
		.fill(0)
		.map(() => Math.random() * 2 - 1);

	// Decoder weights (D → N)
	let decoderWeights: number[][] = Array(N)
		.fill(0)
		.map(() =>
			Array(D)
				.fill(0)
				.map(() => Math.random() * 2 - 1)
		);

	// Compute activations
	function computeActivations() {
		// Decode: hidden → neuronal
		neuronActivations = decoderWeights.map((weights) => {
			let sum = 0;
			for (let i = 0; i < D; i++) {
				sum += weights[i] * hiddenState[i];
			}
			return sum;
		});

		// Apply ReLU with threshold
		sparseActivations = neuronActivations.map((act) => Math.max(0, act - thresholdValue));

		// Calculate sparsity (percentage of zero activations)
		const activeCount = sparseActivations.filter((act) => act > 0).length;
		sparsity = ((N - activeCount) / N) * 100;
	}

	// Animation loop
	function animate() {
		if (!isPlaying) return;

		time += 0.02;

		// Simulate input changes
		hiddenState = hiddenState.map((val, i) => Math.sin(time + i) * Math.cos(time * 0.5 + i * 0.3));

		computeActivations();

		animationFrame = requestAnimationFrame(animate);
	}

	function togglePlay() {
		isPlaying = !isPlaying;
		if (isPlaying) {
			animate();
		} else {
			cancelAnimationFrame(animationFrame);
		}
	}

	function reset() {
		isPlaying = false;
		cancelAnimationFrame(animationFrame);
		time = 0;
		hiddenState = Array(D)
			.fill(0)
			.map(() => Math.random() * 2 - 1);
		computeActivations();
	}

	function updateThreshold(value: number) {
		thresholdValue = value;
		threshold.set(value);
		computeActivations();
	}

	function nextStep() {
		currentStep = (currentStep + 1) % 5;
		reset();
	}

	function prevStep() {
		currentStep = (currentStep - 1 + 5) % 5;
		reset();
	}

	onMount(() => {
		computeActivations();
		return () => {
			if (animationFrame) cancelAnimationFrame(animationFrame);
		};
	});

	// Steps for the explanation
	const steps = [
		{
			title: 'The Dense Problem',
			description:
				'Standard neural networks have dense activations - almost every neuron fires for every input, making them hard to interpret.'
		},
		{
			title: 'Information Bottleneck',
			description:
				'We force information through a narrow "hidden" dimension (D=256) before expanding to a larger "neuronal" space (N=1024).'
		},
		{
			title: 'Sparse Activation',
			description:
				'By applying ReLU with a learnable threshold, we ensure only ~5-14% of neurons activate for any given input.'
		},
		{
			title: 'Monosemantic Neurons',
			description:
				'Sparse activation forces neurons to specialize for specific features - like a neuron that only fires for spaces or uppercase letters!'
		},
		{
			title: 'Low-Rank Structure',
			description:
				'The transformation D→N→D has low rank (≤D), creating an overcomplete dictionary for sparse coding.'
		}
	];

	// Blue-only palette for animation: inactive = blue-gray, active = primary cyan shades
	function getNeuronColor(activation: number, maxActivation: number): string {
		if (activation === 0) return 'hsl(215, 18%, 38%)'; // subtle blue-gray inactive
		const intensity = Math.min(activation / (maxActivation * 0.8), 1);
		const lightness = 52 - intensity * 22;
		return `hsl(186, 85%, ${Math.max(32, lightness)}%)`; // primary blue
	}
</script>

<svelte:head>
	<title>Understanding ReLU Low-Rank - Sparse Neural Representations</title>
</svelte:head>

<div class="max-w-6xl mx-auto px-4 py-8">
	<!-- Header -->
	<div class="mb-8">
		<a href="/" class="text-xs text-primary hover:text-primary/80 mb-4 inline-block font-mono tracking-[0.18em] uppercase">
			← Back to Visualization
		</a>
		<h1 class="text-4xl font-serif text-foreground mb-4">
			Understanding ReLU Low-Rank: Sparse Neural Representations
		</h1>
		<p class="text-lg text-muted-foreground">
			How information bottlenecks create interpretable, sparse, monosemantic neurons
		</p>
	</div>

	<!-- Progress Steps -->
	<div class="mb-8 flex justify-between items-center">
		<button
			on:click={prevStep}
			class="px-4 py-2 rounded-md bg-surface/80 text-muted-foreground border border-border/25 hover:bg-surface-hover/90 transition-colors text-xs font-mono"
		>
			← Previous
		</button>
		<div class="flex gap-2">
			{#each steps as _, i}
				<button
					on:click={() => {
						currentStep = i;
						reset();
					}}
					class="w-3 h-3 rounded-full transition-all"
					class:bg-neon-cyan={currentStep === i}
					class:bg-secondary={currentStep !== i}
				/>
			{/each}
		</div>
		<button
			on:click={nextStep}
			class="px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors text-xs font-mono"
		>
			Next →
		</button>
	</div>

	<!-- Main Content: Animation full width, then explanation boxes below -->
	<div class="flex flex-col gap-8 mb-8">
		<div class="animation-block viz-glass p-5 w-full">
			<div class="grid grid-cols-1 md:grid-cols-[1fr_280px] gap-8 items-center">
				<!-- Left: Animation (SVG) -->
				<div class="min-w-0">
					<!-- Network Visualization -->
					<div class="rounded-xl overflow-hidden bg-transparent flex items-center justify-center">
						<svg viewBox="0 0 400 620" class="w-full block"
					style="height: min(580px, calc(100vh - 200px)); background: transparent;"
					preserveAspectRatio="xMidYMid meet"
				>
				<defs>
					<filter id="circleGlow" x="-50%" y="-50%" width="200%" height="200%">
						<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
						<feMerge>
							<feMergeNode in="blur" />
							<feMergeNode in="SourceGraphic" />
						</feMerge>
					</filter>
				</defs>
				<!-- Hidden Layer (D=4): title → subtitle → gap 28 → circles -->
				<g>
					<text x="200" y="26" text-anchor="middle" font-size="14" font-weight="600" fill="hsl(210 40% 92%)">
						Hidden Space (D={D})
					</text>
					<text x="200" y="44" text-anchor="middle" font-size="11" fill="hsl(215 20% 52%)">
						Compact, dense representation
					</text>

					{#each hiddenState as value, i}
						{@const x = 100 + i * 60}
						{@const y = 90}
						{@const normalizedValue = Math.max(-1, Math.min(1, value))}
						{@const color =
							normalizedValue > 0
								? `hsl(186, 78%, ${56 - normalizedValue * 32}%)`
								: `hsl(220, 25%, ${28 + (1 + normalizedValue) * 8}%)`}

						<circle cx={x} cy={y} r="28" fill={color} stroke="hsl(240 28% 12%)" stroke-width="1.5" filter="url(#circleGlow)" />
						<text x={x} y={y + 5} text-anchor="middle" font-size="10" fill="hsl(240 33% 8%)" font-weight="600">
							{normalizedValue.toFixed(2)}
						</text>
					{/each}
				</g>

				<!-- Decoder Arrow: gap 28 below circles, arrow 30 tall -->
				<g>
					<line x1="200" y1="136" x2="200" y2="166" stroke="hsl(215 18% 42%)" stroke-width="2" />
					<polygon points="200,166 195,156 205,156" fill="hsl(215 18% 42%)" />
					<text x="210" y="156" font-size="12" fill="hsl(215 18% 52%)">Decode (D→N)</text>
				</g>

				<!-- Neuronal Layer (N=8): gap 28 → title → subtitle → gap 28 → circles -->
				<g>
					<text x="200" y="194" text-anchor="middle" font-size="14" font-weight="600" fill="hsl(210 40% 92%)">
						Neuronal Space (N={N})
					</text>
					<text x="200" y="212" text-anchor="middle" font-size="11" fill="hsl(215 20% 52%)">
						Before threshold (raw activations)
					</text>

					{#each neuronActivations as value, i}
						{@const x = 50 + (i % 4) * 100}
						{@const y = 260 + Math.floor(i / 4) * 84}
						{@const maxAbs = Math.max(...neuronActivations.map(Math.abs))}
						{@const normalizedValue = Math.max(-1, Math.min(1, value / maxAbs))}
						{@const color =
							normalizedValue > 0
								? `hsl(186, 78%, ${58 - normalizedValue * 32}%)`
								: `hsl(220, 22%, 26%)`}

						<circle cx={x} cy={y} r="30" fill={color} stroke="hsl(240 28% 12%)" stroke-width="1.5" filter="url(#circleGlow)" />
						<text x={x} y={y + 5} text-anchor="middle" font-size="9" fill="hsl(210 40% 92%)" font-weight="600">
							{value.toFixed(2)}
						</text>

						<!-- Threshold line (below threshold = muted blue) -->
						{#if value < thresholdValue}
							<line
								x1={x - 24}
								y1={y}
								x2={x + 24}
								y2={y}
								stroke="hsl(215 25% 45%)"
								stroke-width="2"
								stroke-dasharray="4 3"
								opacity="0.85"
							/>
						{/if}
					{/each}
				</g>

				<!-- ReLU + Threshold: gap 28 below neuronal circles, arrow 30 tall -->
				<g>
					<line x1="200" y1="368" x2="200" y2="398" stroke="hsl(215 18% 42%)" stroke-width="2" />
					<polygon points="200,398 195,388 205,388" fill="hsl(215 18% 42%)" />
					<text x="210" y="388" font-size="12" fill="hsl(186 85% 55%)" font-weight="600">
						ReLU(x − τ)
					</text>
				</g>

				<!-- Sparse Neuronal Layer: gap 28 → title → subtitle → gap 28 → circles -->
				<g>
					<text x="200" y="426" text-anchor="middle" font-size="14" font-weight="600" fill="hsl(210 40% 92%)">
						Sparse Activations
					</text>
					<text x="200" y="444" text-anchor="middle" font-size="11" fill="hsl(186 85% 55%)">
						Sparsity: {sparsity.toFixed(1)}% inactive
					</text>

					{#each sparseActivations as value, i}
						{@const x = 50 + (i % 4) * 100}
						{@const y = 492 + Math.floor(i / 4) * 84}
						{@const maxActivation = Math.max(...sparseActivations)}
						{@const color = getNeuronColor(value, maxActivation)}

						<circle cx={x} cy={y} r="30" fill={color} stroke="hsl(240 28% 12%)" stroke-width="1.5" filter="url(#circleGlow)" />
						<text
							x={x}
							y={y + 5}
							text-anchor="middle"
							font-size="9"
							fill={value > 0 ? 'hsl(240 33% 8%)' : 'hsl(215 18% 42%)'}
							font-weight="600"
						>
							{value > 0 ? value.toFixed(2) : '0'}
						</text>

						<!-- Active indicator -->
						{#if value > 0}
							<circle cx={x + 30} cy={y - 30} r="6" fill="hsl(186 85% 55%)" filter="url(#circleGlow)" />
						{/if}
					{/each}
				</g>
						</svg>
					</div>

					<p class="text-xs text-muted-foreground mt-5 text-center">Hidden (D) → Decode → Neuronal (N) → ReLU(·−τ) → Sparse</p>
				</div>

				<!-- Right: Current Step above controls -->
				<div class="flex flex-col gap-6 w-full md:w-[280px]">
					<!-- Current Step -->
					<div class="blend-card p-4 rounded-lg">
						<div class="flex items-center gap-2 mb-1.5">
							<span class="text-xs font-medium text-primary">Step {currentStep + 1} of 5</span>
						</div>
						<h2 class="text-base font-serif font-semibold mb-2 text-foreground">{steps[currentStep].title}</h2>
						<p class="text-xs text-muted-foreground leading-relaxed">{steps[currentStep].description}</p>
					</div>

					<!-- Controls box (Play, Reset, threshold, Live stats) -->
					<div class="blend-card rounded-lg p-5 space-y-8">
					<div class="flex gap-3">
						<button
							on:click={togglePlay}
							class="relu-play-btn flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all"
						>
							{isPlaying ? '⏸ Pause' : '▶ Play'}
						</button>
						<button
							on:click={reset}
							class="px-4 py-2.5 rounded-lg bg-surface/60 text-muted-foreground border border-border/25 hover:bg-surface-hover/80 hover:text-foreground transition-all text-sm font-medium"
						>
							Reset
						</button>
					</div>

					<div>
						<div class="flex justify-between items-baseline mb-2">
							<label class="text-sm font-medium text-foreground" for="threshold-slider">Threshold τ</label>
							<span class="text-sm font-mono text-primary tabular-nums">{thresholdValue.toFixed(2)}</span>
						</div>
						<input
							id="threshold-slider"
							type="range"
							min="0"
							max="2"
							step="0.1"
							value={thresholdValue}
							on:input={(e) => updateThreshold(parseFloat(e.currentTarget.value))}
							class="relu-threshold-slider w-full h-2 rounded-full appearance-none cursor-pointer"
						/>
						<div class="text-xs text-muted-foreground mt-1.5 flex justify-between">
							<span>More active</span>
							<span>More sparse</span>
						</div>
					</div>

					<div class="rounded-lg py-3 bg-transparent">
						<div class="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Live stats</div>
						<div class="flex flex-col gap-1 text-sm">
							<div><span class="text-muted-foreground">Active</span> <span class="font-mono font-semibold text-foreground tabular-nums">{sparseActivations.filter((a) => a > 0).length}/{N}</span></div>
							<div><span class="text-muted-foreground">Sparsity</span> <span class="font-mono font-semibold text-foreground tabular-nums">{sparsity.toFixed(1)}%</span></div>
							<div><span class="text-muted-foreground">Silent</span> <span class="font-mono font-semibold text-foreground tabular-nums">{(((N - sparseActivations.filter((a) => a > 0).length) / N) * 100).toFixed(1)}%</span></div>
						</div>
					</div>
				</div>
				</div>
			</div>
		</div>
	</div>

	<!-- Mathematics and Key Insights (below animation) -->
	<div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 [&>*]:min-w-0">
		<div class="blend-card p-4 rounded-lg">
			<h3 class="text-sm font-serif font-semibold tracking-tight text-foreground mb-3">The Mathematics</h3>
			<div class="space-y-3 text-xs">
				<div>
					<p class="font-medium text-primary mb-1">1. Decode to Neuronal Space</p>
					<code class="block blend-card text-foreground px-3 py-2 rounded-lg mt-1 text-[11px] font-mono">
						x_raw = LayerNorm(v) × W_decoder<br />
						where v ∈ ℝ^D, x_raw ∈ ℝ^N
					</code>
				</div>
				<div>
					<p class="font-medium text-primary mb-1">2. Apply Sparse Activation</p>
					<code class="block blend-card text-foreground px-3 py-2 rounded-lg mt-1 text-[11px] font-mono">
						x_sparse = ReLU(x_raw - τ)<br />
						= max(0, x_raw - τ)
					</code>
				</div>
				<div>
					<p class="font-medium text-primary mb-1">3. Encode Back</p>
					<code class="block blend-card text-foreground px-3 py-2 rounded-lg mt-1 text-[11px] font-mono">
						v_new = x_sparse × W_encoder<br />
						where v_new ∈ ℝ^D
					</code>
				</div>
				<div>
					<p class="font-medium text-primary mb-1">4. Effective Rank</p>
					<code class="block blend-card text-foreground px-3 py-2 rounded-lg mt-1 text-[11px] font-mono">
						rank(W_encoder × W_decoder) ≤ min(D, N) = D
					</code>
				</div>
			</div>
		</div>
		<div class="blend-card p-4 rounded-lg flex flex-col gap-4">
			<div>
				<h3 class="text-sm font-serif font-semibold tracking-tight text-foreground mb-3">Key Insights</h3>
				<ul class="space-y-2 text-xs text-foreground leading-relaxed">
					<li class="flex items-start gap-2">
						<span class="text-primary mt-1">✓</span>
						<span>Only 5-14% of neurons active → sparse, interpretable representations</span>
					</li>
					<li class="flex items-start gap-2">
						<span class="text-primary mt-1">✓</span>
						<span>Information bottleneck forces neurons to specialize</span>
					</li>
					<li class="flex items-start gap-2">
						<span class="text-primary mt-1">✓</span>
						<span>Learnable threshold τ adapts during training</span>
					</li>
				</ul>
			</div>
			<div>
				<h3 class="text-sm font-serif font-semibold tracking-tight text-foreground mb-2">The Polysemanticity Problem</h3>
				<div class="space-y-2 text-sm text-muted-foreground leading-relaxed">
					<p>
						In traditional neural networks, neurons are <em>polysemantic</em> — they respond to many
						unrelated features. This makes understanding what the network has learned extremely
						difficult.
					</p>
					<p>
						This happens because networks try to pack as many features as possible into limited
						neurons through <em>superposition</em> — representing multiple features in overlapping
						patterns.
					</p>
				</div>
			</div>
		</div>
	</div>

	<!-- Detailed Explanation Sections -->
	<div class="space-y-10">
		<!-- Section 2: The Solution -->
		<section class="section-card blend-card p-6 md:p-8">
			<h2 class="section-title text-xl font-serif font-semibold tracking-tight text-foreground mb-6">The Low-Rank Solution: Force Specialization</h2>
			<div class="prose max-w-none">
				<p class="text-sm text-muted-foreground leading-relaxed mb-5">
					The key insight: by forcing information through a narrow bottleneck, then expanding to a
					larger sparse space, we can create <em>monosemantic</em> neurons that respond to just one
					interpretable feature.
				</p>

				<div class="grid md:grid-cols-3 gap-4 my-6">
					<div class="blend-card rounded-lg p-5 border border-border/20">
						<p class="text-xs font-medium text-primary uppercase tracking-wider mb-2">Step 1</p>
						<h4 class="font-semibold text-foreground mb-2">Compress</h4>
						<p class="text-sm text-muted-foreground leading-relaxed">
							Information flows through narrow hidden dimension D=256. Forces essential features only.
						</p>
					</div>
					<div class="blend-card rounded-lg p-5 border border-border/20">
						<p class="text-xs font-medium text-primary uppercase tracking-wider mb-2">Step 2</p>
						<h4 class="font-semibold text-foreground mb-2">Expand</h4>
						<p class="text-sm text-muted-foreground leading-relaxed">
							Decode to larger neuronal space N=1024. Creates overcomplete dictionary.
						</p>
					</div>
					<div class="blend-card rounded-lg p-5 border border-border/20">
						<p class="text-xs font-medium text-primary uppercase tracking-wider mb-2">Step 3</p>
						<h4 class="font-semibold text-foreground mb-2">Sparsify</h4>
						<p class="text-sm text-muted-foreground leading-relaxed">
							Apply ReLU with threshold. Only top neurons activate.
						</p>
					</div>
				</div>

				<div class="callout blend-card rounded-lg p-5 mt-5 border-l-4 border-primary/50">
					<p class="text-xs font-medium text-primary uppercase tracking-wider mb-2">Result</p>
					<h4 class="font-semibold text-foreground mb-2">Monosemantic Neurons</h4>
					<p class="text-sm text-muted-foreground leading-relaxed">
						Real example from our model: Neuron #231 in layer 7 fires specifically for space
						characters with 85% selectivity. Neuron #92 specializes in uppercase 'U'. Each neuron
						has a clear, interpretable function.
					</p>
				</div>
			</div>
		</section>

		<!-- Section 4: The Learnable Threshold -->
		<section class="section-card blend-card p-6 md:p-8">
			<h2 class="section-title text-xl font-serif font-semibold tracking-tight text-foreground mb-6">The Learnable Threshold: Dynamic Sparsity</h2>
			<div class="prose max-w-none">
				<p class="text-sm text-muted-foreground leading-relaxed mb-5">
					The threshold τ is not fixed — it's a learnable parameter that adapts during training to
					find the optimal sparsity level.
				</p>

				<div class="overflow-x-auto my-6 rounded-lg overflow-hidden">
					<table class="min-w-full border-collapse">
						<thead class="bg-transparent">
							<tr>
								<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Training Phase</th>
								<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Threshold τ</th>
								<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Sparsity</th>
								<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Interpretation</th>
							</tr>
						</thead>
						<tbody class="text-sm text-muted-foreground divide-y divide-border/10">
							<tr><td class="px-4 py-3">Initialization</td><td class="px-4 py-3 font-mono tabular-nums">0.50</td><td class="px-4 py-3">~15%</td><td class="px-4 py-3">Model needs capacity to explore</td></tr>
							<tr><td class="px-4 py-3">Early Training</td><td class="px-4 py-3 font-mono tabular-nums">0.42</td><td class="px-4 py-3">~12%</td><td class="px-4 py-3">Learning basic patterns</td></tr>
							<tr><td class="px-4 py-3">Mid Training</td><td class="px-4 py-3 font-mono tabular-nums">0.58</td><td class="px-4 py-3">~8%</td><td class="px-4 py-3">Features beginning to specialize</td></tr>
							<tr><td class="px-4 py-3">Late Training</td><td class="px-4 py-3 font-mono tabular-nums">0.72</td><td class="px-4 py-3">~5%</td><td class="px-4 py-3">Specialist neurons emerge</td></tr>
							<tr><td class="px-4 py-3">Convergence</td><td class="px-4 py-3 font-mono tabular-nums">0.85</td><td class="px-4 py-3">~5%</td><td class="px-4 py-3">Stable sparse solution</td></tr>
						</tbody>
					</table>
				</div>

				<p class="text-sm text-muted-foreground leading-relaxed">
					The gradient of the threshold tells the model whether to increase sparsity (raise τ) or
					allow more neurons to activate (lower τ), finding the sweet spot for performance and
					interpretability.
				</p>
			</div>
		</section>

		<!-- Section 7: Comparison -->
		<section class="section-card blend-card p-6 md:p-8">
			<h2 class="section-title text-xl font-serif font-semibold tracking-tight text-foreground mb-6">Comparison: Standard MLP vs Low-Rank ReLU</h2>
			<div class="overflow-x-auto rounded-lg overflow-hidden">
				<table class="min-w-full border-collapse">
					<thead class="bg-transparent">
						<tr>
							<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Aspect</th>
							<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Standard Transformer MLP</th>
							<th class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">BDH Low-Rank ReLU</th>
						</tr>
					</thead>
					<tbody class="text-sm text-muted-foreground divide-y divide-border/10">
						<tr><td class="px-4 py-3 font-medium text-foreground">Activation Pattern</td><td class="px-4 py-3">Dense (~100% active with GELU)</td><td class="px-4 py-3"><span class="text-primary font-semibold">Sparse (~5-14% active)</span></td></tr>
						<tr><td class="px-4 py-3 font-medium text-foreground">Neuron Type</td><td class="px-4 py-3">Polysemantic (multiple features per neuron)</td><td class="px-4 py-3"><span class="text-primary font-semibold">Monosemantic (one feature per neuron)</span></td></tr>
						<tr><td class="px-4 py-3 font-medium text-foreground">Interpretability</td><td class="px-4 py-3">Low (black box)</td><td class="px-4 py-3"><span class="text-primary font-semibold">High (clear feature mapping)</span></td></tr>
						<tr><td class="px-4 py-3 font-medium text-foreground">Weight Sharing</td><td class="px-4 py-3">Per-layer weights</td><td class="px-4 py-3"><span class="text-primary font-semibold">Global shared encoder/decoder</span></td></tr>
						<tr><td class="px-4 py-3 font-medium text-foreground">Parameters (8 layers)</td><td class="px-4 py-3">~4.2M</td><td class="px-4 py-3"><span class="text-primary font-semibold">~786k (81% reduction)</span></td></tr>
						<tr><td class="px-4 py-3 font-medium text-foreground">Biological Plausibility</td><td class="px-4 py-3">Low (arbitrary activations)</td><td class="px-4 py-3"><span class="text-primary font-semibold">High (sparse, positive)</span></td></tr>
					</tbody>
				</table>
			</div>
		</section>
	</div>

	<!-- Navigation -->
	<div class="mt-12 flex justify-between items-center py-6 border-t border-border/25">
		<a
			href="/rope"
			class="px-6 py-3 rounded-md bg-surface/80 text-muted-foreground border border-border/25 hover:bg-surface-hover/90 transition-colors text-xs font-mono"
		>
			← Previous: ROPE
		</a>
		<a
			href="/"
			class="px-6 py-3 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors text-xs font-mono"
		>
			Back to Visualization
		</a>
	</div>
</div>

<style>
	/* Interactive Animation box: glassmorphism */
	.viz-glass {
		backdrop-filter: blur(8px);
		-webkit-backdrop-filter: blur(8px);
		border: 1px solid rgba(255, 255, 255, 0.1);
		background: rgba(11, 17, 23, 0.5);
		border-radius: var(--radius);
	}

	/* Play button: outlined #1F6FEB with hover glow */
	.relu-play-btn {
		background: transparent;
		color: #1F6FEB;
		border: 1.5px solid #1F6FEB;
	}
	.relu-play-btn:hover {
		background: rgba(31, 111, 235, 0.15);
		box-shadow: 0 0 12px rgba(31, 111, 235, 0.4);
	}

	/* Boxes that blend with the theme — transparent surface, blur, theme borders */
	:global(.blend-card) {
		background: hsl(var(--surface) / 0.28);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border: 1px solid hsl(var(--border) / 0.2);
		border-radius: var(--radius);
		box-shadow: 0 1px 0 0 hsl(var(--background) / 0.5);
	}

	/* Inner boxes: no visible box — blend into parent; callouts keep border-l-4 from Tailwind */
	:global(.blend-card .blend-card),
	:global(.section-card .blend-card) {
		background: transparent;
		backdrop-filter: none;
		-webkit-backdrop-filter: none;
		border: none;
		box-shadow: none;
	}
	/* Code blocks inside cards: no box, just text */
	:global(.blend-card code.blend-card) {
		background: transparent;
		border: none;
	}

	/* Professional threshold slider */
	.relu-threshold-slider {
		background: hsl(var(--muted));
	}
	.relu-threshold-slider::-webkit-slider-thumb {
		appearance: none;
		width: 18px;
		height: 18px;
		background: hsl(var(--primary));
		border-radius: 50%;
		border: 2px solid hsl(var(--surface));
		cursor: pointer;
		box-shadow: 0 0 0 1px hsl(var(--border) / 0.5);
	}
	.relu-threshold-slider::-webkit-slider-runnable-track {
		height: 8px;
		border-radius: 4px;
		background: hsl(var(--muted));
	}
	.relu-threshold-slider::-moz-range-thumb {
		width: 18px;
		height: 18px;
		background: hsl(var(--primary));
		border-radius: 50%;
		border: 2px solid hsl(var(--surface));
		cursor: pointer;
		box-shadow: 0 0 0 1px hsl(var(--border) / 0.5);
	}
	.relu-threshold-slider::-moz-range-track {
		height: 8px;
		border-radius: 4px;
		background: hsl(var(--muted));
	}

	code {
		font-family: 'JetBrains Mono', monospace;
	}
</style>
