<script lang="ts">
	import { onMount } from 'svelte';

	const API_URL = '';

	interface AtlasEntry {
		concept: string;
		triggers: string[];
		score: number;
	}

	let atlas: Record<string, AtlasEntry> = {};
	let isLoading = true;
	let error = '';
	let searchQuery = '';
	let viewMode: 'grid' | 'constellation' = 'constellation';
	let hoveredConcept: string | null = null;

	// Concept color palette
	const conceptColors: Record<string, string> = {};
	const palette = [
		'#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
		'#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6',
		'#a855f7', '#6366f1', '#0ea5e9', '#84cc16', '#f59e0b',
	];

	function displayNeuronId(id: string): string {
		return id.startsWith('N') ? id : `N${id}`;
	}

	function getConceptColor(concept: string): string {
		if (!conceptColors[concept]) {
			const idx = Object.keys(conceptColors).length % palette.length;
			conceptColors[concept] = palette[idx];
		}
		return conceptColors[concept];
	}

	// Jaccard similarity between concepts based on triggers
	function conceptSimilarity(a: string[], b: string[]): number {
		if (a.length === 0 && b.length === 0) return 0;
		const sa = new Set(a.map((t) => t.toLowerCase()));
		const sb = new Set(b.map((t) => t.toLowerCase()));
		let inter = 0;
		sa.forEach((t) => {
			if (sb.has(t)) inter++;
		});
		const union = sa.size + sb.size - inter;
		return union > 0 ? inter / union : 0;
	}

	async function loadAtlas() {
		try {
			const res = await fetch(`${API_URL}/api/activation-atlas`);
			if (!res.ok) throw new Error(`API error: ${res.status}`);
			const data = await res.json();
			atlas = data.atlas;
			isLoading = false;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load atlas';
			isLoading = false;
		}
	}

	onMount(loadAtlas);

	// Group neurons by concept
	$: conceptGroups = Object.entries(atlas).reduce(
		(acc, [neuronId, entry]) => {
			if (!acc[entry.concept]) acc[entry.concept] = [];
			acc[entry.concept].push({ neuronId, ...entry });
			return acc;
		},
		{} as Record<string, Array<{ neuronId: string } & AtlasEntry>>
	);

	$: filteredConcepts = Object.keys(conceptGroups)
		.filter((c) => searchQuery === '' || c.toLowerCase().includes(searchQuery.toLowerCase()))
		.sort((a, b) => conceptGroups[b].length - conceptGroups[a].length);

	$: totalNeurons = Object.keys(atlas).length;

	// Constellation layout: concept nodes + neuron clusters (expanded for interpretability)
	$: layout = (() => {
		const concepts = filteredConcepts;
		const n = concepts.length;
		const angleStep = (2 * Math.PI) / Math.max(n, 1);
		const cx = 800;
		const cy = 600;
		const radius = 450;

		const conceptPos: Record<string, { x: number; y: number }> = {};
		concepts.forEach((concept, i) => {
			const angle = i * angleStep - Math.PI / 2;
			conceptPos[concept] = {
				x: cx + Math.cos(angle) * radius,
				y: cy + Math.sin(angle) * radius,
			};
		});

		const neurons: Array<{ x: number; y: number; concept: string; neuronId: string; score: number; color: string }> = [];
		concepts.forEach((concept) => {
			const pos = conceptPos[concept];
			const neuronsList = conceptGroups[concept];
			const spread = Math.min(100, neuronsList.length * 12);

			neuronsList.forEach((n, ni) => {
				const subAngle = (ni / neuronsList.length) * 2 * Math.PI;
				neurons.push({
					x: pos.x + Math.cos(subAngle) * spread,
					y: pos.y + Math.sin(subAngle) * spread,
					concept,
					neuronId: n.neuronId,
					score: n.score,
					color: getConceptColor(concept),
				});
			});
		});

		// Edges between concepts (similarity)
		const edges: Array<{ c1: string; c2: string; sim: number }> = [];
		for (let i = 0; i < concepts.length; i++) {
			for (let j = i + 1; j < concepts.length; j++) {
				const t1 = conceptGroups[concepts[i]][0]?.triggers ?? [];
				const t2 = conceptGroups[concepts[j]][0]?.triggers ?? [];
				const sim = conceptSimilarity(t1, t2);
				if (sim > 0.05) edges.push({ c1: concepts[i], c2: concepts[j], sim });
			}
		}

		return { conceptPos, neurons, edges, concepts };
	})();
</script>

<svelte:head>
	<title>Activation Atlas - BDH Visualization</title>
</svelte:head>

<div class="h-screen overflow-hidden flex flex-col">
	<div class="w-full max-w-[1920px] mx-auto px-6 py-4 flex-1 flex flex-col min-h-0">
	{#if isLoading}
		<div class="flex-1 flex items-center justify-center min-h-0">
			<div class="text-center">
				<div class="spinner mx-auto mb-4"></div>
				<p class="text-muted-foreground">Loading activation atlas...</p>
			</div>
		</div>
	{:else if error}
		<div class="flex-1 flex items-center justify-center min-h-0">
		<div class="rounded-xl p-6 bg-red-500/10 border border-red-500/40 text-center max-w-md">
			<p class="text-red-400 mb-2">{error}</p>
			<p class="text-sm text-red-400/80">Make sure the backend is running: <code class="px-2 py-1 rounded bg-red-500/20 font-mono">python main.py</code></p>
		</div>
		</div>
	{:else}
		{#if viewMode === 'constellation'}
			<!-- Constellation View -->
			<div
				class="relative rounded-xl overflow-hidden flex-1 min-h-0 flex flex-col bg-background"
			>
				<!-- Floating box: title and description -->
				<div class="absolute top-4 left-4 z-10 rounded-xl p-4 bg-surface/60 border border-border/50 shadow-xl max-w-sm backdrop-blur-sm">
					<h1 class="text-xl font-serif text-foreground mb-1">Activation Atlas</h1>
					<p class="text-sm text-muted-foreground">
						Each neuron in the sparse network specializes for a semantic concept.
						This atlas maps neuron IDs to their learned specializations.
					</p>
				</div>
				<!-- Stats badge top-right -->
				<div class="absolute top-4 right-4 z-10 px-3 py-1.5 rounded-lg text-xs font-mono bg-surface/80 border border-border/60 text-muted-foreground">
					{totalNeurons} neurons mapped to {layout.concepts.length} concepts
				</div>

				<svg
					viewBox="0 0 1600 1200"
					class="w-full h-full min-h-0"
					preserveAspectRatio="xMidYMid meet"
				>
					<!-- Inter-concept edges with similarity labels -->
					{#each layout.edges as edge}
						{@const p1 = layout.conceptPos[edge.c1]}
						{@const p2 = layout.conceptPos[edge.c2]}
						{@const mx = (p1.x + p2.x) / 2}
						{@const my = (p1.y + p2.y) / 2}
						<line
							x1={p1.x}
							y1={p1.y}
							x2={p2.x}
							y2={p2.y}
							stroke={getConceptColor(edge.c1)}
							stroke-width="2"
							opacity="0.4"
						/>
						<circle cx={mx} cy={my} r="18" style="fill: hsl(var(--background));" />
						<text x={mx} y={my + 6} text-anchor="middle" font-size="12" fill="#94a3b8" font-family="monospace">
							{edge.sim.toFixed(2)}
						</text>
					{/each}

					<!-- Neuron-to-concept connections -->
					{#each layout.neurons as node}
						{@const pos = layout.conceptPos[node.concept]}
						<line
							x1={pos.x}
							y1={pos.y}
							x2={node.x}
							y2={node.y}
							stroke={node.color}
							stroke-width="1"
							opacity="0.25"
						/>
					{/each}

					<!-- Neuron dots -->
					{#each layout.neurons as node}
						<circle
							cx={node.x}
							cy={node.y}
							r={Math.max(5, Math.min(14, node.score * 10))}
							fill={node.color}
							opacity="0.8"
							class="cursor-default"
						/>
					{/each}

					<!-- Concept nodes (large circles with labels) -->
					{#each layout.concepts as concept}
						{@const pos = layout.conceptPos[concept]}
						{@const color = getConceptColor(concept)}
						{@const isHovered = hoveredConcept === concept}
						<g
							on:mouseenter={() => (hoveredConcept = concept)}
							on:mouseleave={() => (hoveredConcept = null)}
							class="cursor-pointer"
						>
							<!-- Glow -->
							<circle cx={pos.x} cy={pos.y} r="56" fill={color} opacity={isHovered ? 0.25 : 0.1} />
							<!-- Concept circle -->
							<circle
								cx={pos.x}
								cy={pos.y}
								r="42"
								fill={color}
								opacity={isHovered ? 1 : 0.85}
								stroke={isHovered ? '#fff' : 'transparent'}
								stroke-width="3"
							/>
							<!-- Label -->
							<text
								x={pos.x}
								y={pos.y + 7}
								text-anchor="middle"
								font-size="16"
								font-weight="600"
								fill="#fff"
							>
								{concept}
							</text>
						</g>
					{/each}
				</svg>

				<!-- Hover popup -->
				{#if hoveredConcept}
					{@const entry = conceptGroups[hoveredConcept]?.[0]}
					{@const neurons = conceptGroups[hoveredConcept] ?? []}
					{@const color = getConceptColor(hoveredConcept)}
					<div
						class="absolute rounded-xl p-5 border-2 z-20 min-w-[280px]"
						style="
							top: 50%;
							right: 40px;
							transform: translateY(-50%);
							background: rgba(15, 23, 42, 0.9);
							border-color: {color};
							box-shadow: 0 0 24px {color}40;
						"
					>
						<div class="flex items-center gap-2 mb-4">
							<div
								class="w-4 h-4 rounded-full flex-shrink-0"
								style="background: {color}; box-shadow: 0 0 10px {color};"
							></div>
							<span class="font-semibold text-base text-foreground capitalize">{hoveredConcept}</span>
						</div>
						<div class="mb-3">
							<div class="text-sm font-medium text-muted-foreground mb-2">Top Neurons</div>
							<div class="flex flex-wrap gap-2">
								{#each neurons.slice(0, 12) as n}
									<span
										class="px-2 py-1 text-sm font-mono rounded cursor-default"
										style="background: {color}30; color: {color};"
									>
										{displayNeuronId(n.neuronId)}
									</span>
								{/each}
								{#if neurons.length > 12}
									<span class="text-sm text-muted-foreground">+{neurons.length - 12}</span>
								{/if}
							</div>
						</div>
						<div>
							<div class="text-sm font-medium text-muted-foreground mb-2">Triggers</div>
							<p class="text-base text-foreground italic">
								{entry ? entry.triggers.join(', ') : '—'}
							</p>
						</div>
					</div>
				{/if}
			</div>
		{:else}
			<!-- Grid View -->
			<div class="relative rounded-xl overflow-hidden flex-1 min-h-0 flex flex-col bg-background">
				<!-- Floating box: title and description -->
				<div class="absolute top-4 left-4 z-10 rounded-xl p-4 bg-surface/60 border border-border/50 shadow-xl max-w-sm backdrop-blur-sm">
					<h1 class="text-xl font-serif text-foreground mb-1">Activation Atlas</h1>
					<p class="text-sm text-muted-foreground">
						Each neuron in the sparse network specializes for a semantic concept.
						This atlas maps neuron IDs to their learned specializations.
					</p>
				</div>
				<div class="atlas-grid-scroll pt-24 px-4 pb-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 overflow-auto flex-1 min-h-0">
				{#each filteredConcepts as concept}
					{@const neurons = conceptGroups[concept]}
					<div class="rounded-xl p-4 bg-surface/60 border border-border/50 hover:border-primary/40 transition-colors">
						<div class="flex items-center gap-2 mb-3">
							<div
								class="w-3 h-3 rounded-full flex-shrink-0"
								style="background: {getConceptColor(concept)};"
							></div>
							<h3 class="font-medium text-sm text-foreground capitalize">{concept}</h3>
							<span class="text-xs text-muted-foreground ml-auto">{neurons.length}</span>
						</div>
						<div class="flex flex-wrap gap-1">
							{#each neurons as n}
								<span
									class="px-1.5 py-0.5 text-xs font-mono rounded"
									style="background: {getConceptColor(concept)}30; color: {getConceptColor(concept)};"
								>
									{displayNeuronId(n.neuronId)}
								</span>
							{/each}
						</div>
						<div class="mt-2 text-xs text-muted-foreground">
							Triggers: {neurons[0]?.triggers.slice(0, 3).join(', ') ?? '—'}
						</div>
					</div>
				{/each}
				</div>
			</div>
		{/if}
	{/if}
	</div>
</div>

<style>
	.spinner {
		@apply w-8 h-8 border-2 border-border rounded-full;
		border-top-color: hsl(var(--primary));
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	/* Hide scrollbar while keeping overflow scroll */
	.atlas-grid-scroll {
		scrollbar-width: none;
		-ms-overflow-style: none;
	}
	.atlas-grid-scroll::-webkit-scrollbar {
		display: none;
	}
</style>
