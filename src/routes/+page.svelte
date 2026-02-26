<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import type { ActivationData } from '$lib/types';
	import { playbackStore } from '$lib/stores/playback';
	import ForceGraph3D from '$lib/components/ForceGraph3D.svelte';
	import TokenTimeline from '$lib/components/TokenTimeline.svelte';
	import CustomInput from '$lib/components/CustomInput.svelte';
	import LayerSelector from '$lib/components/LayerSelector.svelte';
	import NeuronInspector from '$lib/components/NeuronInspector.svelte';
	import Legend from '$lib/components/Legend.svelte';

	let activationData: ActivationData | null = null;
	let isLoading = true;
	let isInferenceLoading = false;
	let error = '';
	let currentSequenceIdx = 0;
	let dataSource: 'precomputed' | 'api' = 'precomputed';

	// Panel minimize state
	let isLeftMinimized = false;
	let isRightMinimized = false;
	let isBottomMinimized = false;

	const API_URL = '';

	// Load pre-computed activation data
	async function loadData() {
		try {
			const response = await fetch('/data/activations.json');
			if (!response.ok) throw new Error('Failed to load activation data');
			activationData = await response.json();
			// Always refresh vocabulary from live API to get correct BPE mappings
			try {
				const vocabRes = await fetch(`${API_URL}/vocabulary`);
				if (vocabRes.ok && activationData) {
					activationData.vocabulary = await vocabRes.json();
				}
			} catch {}
			dataSource = 'precomputed';
			isLoading = false;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
			isLoading = false;
		}
	}

	// Handle custom input submission from API
	function handleCustomInput(event: CustomEvent<{ activationData: ActivationData }>) {
		activationData = event.detail.activationData;
		currentSequenceIdx = 0;
		dataSource = 'api';
		// Reset playback to start
		playbackStore.reset();
		playbackStore.setMaxToken(activationData.sequences[0].tokens.length - 1);
	}

	// Handle API error
	function handleApiError(event: CustomEvent<{ message: string }>) {
		console.error('API Error:', event.detail.message);
		// Keep showing current data, error is displayed in CustomInput
	}

	// Switch back to pre-computed data
	async function loadPrecomputedData() {
		isLoading = true;
		error = '';
		await loadData();
	}

	onMount(() => {
		if (browser) {
			loadData();
		}
	});

	// Get current sequence
	$: currentSequence = activationData?.sequences[currentSequenceIdx];
</script>

<svelte:head>
	<title>BDH Monosemanticity Visualization</title>
</svelte:head>

<div class="visualization-layout">
	{#if isLoading}
		<!-- Loading state -->
		<div class="flex items-center justify-center h-[60vh]">
			<div class="text-center">
				<div class="spinner mx-auto mb-4"></div>
				<p class="text-paper-muted">Loading activation data...</p>
			</div>
		</div>
	{:else if error}
		<!-- Error state -->
		<div class="flex items-center justify-center h-[60vh]">
			<div class="text-center max-w-md">
				<svg class="w-16 h-16 mx-auto mb-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="1.5"
						d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
					/>
				</svg>
				<h2 class="text-lg font-serif mb-2">Failed to Load Data</h2>
				<p class="text-paper-muted mb-4">{error}</p>
				<button
					class="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors"
					on:click={loadData}
				>
					Retry
				</button>
			</div>
		</div>
	{:else if activationData && currentSequence}
		<!-- 3D Graph: full-screen fixed background -->
		<div class="graph-backdrop">
			<ForceGraph3D data={activationData} sequenceIdx={currentSequenceIdx} />
		</div>

		<!-- Floating left panel -->
		<div class="panel-wrapper panel-left" class:minimized={isLeftMinimized}>
			<div class="panel-header">
				<span class="panel-title">Controls</span>
				<button
					class="panel-toggle"
					on:click={() => (isLeftMinimized = !isLeftMinimized)}
					aria-label={isLeftMinimized ? 'Expand panel' : 'Minimize panel'}
				>
					{#if isLeftMinimized}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
					{:else}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
					{/if}
				</button>
			</div>
			<div class="panel-content">
		<div class="floating-panel floating-panel-left space-y-4">
			<div class="floating-panel-section">
				<div class="flex items-center justify-between py-2">
					<span class="text-sm text-muted-foreground">Data Source:</span>
					<div class="flex items-center gap-2">
						<span
							class="px-2 py-0.5 rounded text-xs font-medium"
							class:bg-emerald-500={dataSource === 'api'}
							class:text-emerald-300={dataSource === 'api'}
							class:bg-primary={dataSource === 'precomputed'}
							class:text-primary={dataSource === 'precomputed'}
							style={dataSource === 'api' ? 'background-color: rgba(16, 185, 129, 0.2);' : dataSource === 'precomputed' ? 'background-color: hsl(var(--primary) / 0.2);' : ''}
						>
							{dataSource === 'api' ? 'Live API' : 'Pre-computed'}
						</span>
						{#if dataSource === 'api'}
							<button class="text-xs text-muted-foreground hover:text-foreground underline" on:click={loadPrecomputedData}>Reset</button>
						{/if}
					</div>
				</div>
			</div>
			<CustomInput
				vocabulary={activationData.vocabulary}
				apiUrl={API_URL}
				isLoading={isInferenceLoading}
				on:submit={handleCustomInput}
				on:error={handleApiError}
			/>
			<LayerSelector data={activationData} sequenceIdx={currentSequenceIdx} />
		</div>
			</div>
		</div>
		{#if isLeftMinimized}
			<button class="panel-expand-btn panel-expand-left" on:click={() => (isLeftMinimized = false)} aria-label="Expand left panel">
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
			</button>
		{/if}

		<!-- Floating right panel -->
		<div class="panel-wrapper panel-right" class:minimized={isRightMinimized}>
			<div class="panel-header">
				<span class="panel-title">Inspector</span>
				<button
					class="panel-toggle"
					on:click={() => (isRightMinimized = !isRightMinimized)}
					aria-label={isRightMinimized ? 'Expand panel' : 'Minimize panel'}
				>
					{#if isRightMinimized}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
					{:else}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
					{/if}
				</button>
			</div>
			<div class="panel-content">
		<div class="floating-panel floating-panel-right space-y-4">
			<NeuronInspector />
			<Legend />
			<div class="floating-panel-section">
				<div class="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Model Info</div>
				<div class="space-y-2 text-sm">
					<div class="flex justify-between"><span class="text-muted-foreground">Model</span><span class="font-mono text-foreground">{activationData.metadata.model_name}</span></div>
					<div class="flex justify-between"><span class="text-muted-foreground">Layers</span><span class="font-mono text-foreground">{activationData.metadata.n_layers}</span></div>
					<div class="flex justify-between"><span class="text-muted-foreground">Neurons/Layer</span><span class="font-mono text-foreground">{activationData.metadata.n_neurons}</span></div>
					<div class="flex justify-between"><span class="text-muted-foreground">Vocab Size</span><span class="font-mono text-foreground">{activationData.metadata.vocab_size}</span></div>
					<div class="flex justify-between"><span class="text-muted-foreground">Specialists</span><span class="font-mono text-foreground">{activationData.specialists.length}</span></div>
				</div>
			</div>
		</div>
			</div>
		</div>
		{#if isRightMinimized}
			<button class="panel-expand-btn panel-expand-right" on:click={() => (isRightMinimized = false)} aria-label="Expand right panel">
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
			</button>
		{/if}

		<!-- Floating bottom controls: Playback + Sequence -->
		<div class="panel-wrapper panel-bottom" class:minimized={isBottomMinimized}>
			<div class="panel-header">
				<span class="panel-title">Timeline & Sequence</span>
				<button
					class="panel-toggle"
					on:click={() => (isBottomMinimized = !isBottomMinimized)}
					aria-label={isBottomMinimized ? 'Expand panel' : 'Minimize panel'}
				>
					{#if isBottomMinimized}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7" /></svg>
					{:else}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
					{/if}
				</button>
			</div>
			<div class="panel-content">
		<div class="floating-bottom-controls">
			<TokenTimeline tokens={currentSequence.tokens} vocabulary={activationData.vocabulary} specialists={activationData.specialists} />
			<div class="floating-panel-section mt-3">
				<div class="flex items-center justify-between mb-1">
					<span class="text-xs font-medium text-muted-foreground uppercase tracking-wider">Current Sequence</span>
					{#if dataSource === 'api'}
						<span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-400/40 font-mono">Custom Input</span>
					{/if}
				</div>
				<p class="font-mono text-sm whitespace-pre-wrap text-foreground">{currentSequence.input_text}</p>
			</div>
		</div>
			</div>
		</div>
		{#if isBottomMinimized}
			<button class="panel-expand-btn panel-expand-bottom" on:click={() => (isBottomMinimized = false)} aria-label="Expand bottom panel">
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7" /></svg>
			</button>
		{/if}
	{/if}
</div>

<style>
	.visualization-layout {
		min-height: 100vh;
		position: relative;
	}

	/* 3D Graph: full-screen fixed background */
	.graph-backdrop {
		position: fixed;
		inset: 0;
		z-index: 0;
		width: 100%;
		height: 100%;
	}

	/* Panel wrapper — glassmorphic, smooth transitions */
	.panel-wrapper {
		position: fixed;
		z-index: 10;
		background: rgba(0, 0, 0, 0.5);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border: 1px solid hsl(var(--border) / 0.5);
		border-radius: 0.75rem;
		overflow: hidden;
		transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
	}
	.panel-left {
		top: 80px;
		left: 20px;
		max-width: 350px;
		max-height: calc(100vh - 120px);
	}
	.panel-left.minimized {
		transform: translateX(calc(-100% - 24px));
	}
	.panel-right {
		top: 80px;
		right: 20px;
		max-width: 350px;
		max-height: calc(100vh - 120px);
	}
	.panel-right.minimized {
		transform: translateX(calc(100% + 24px));
	}
	.panel-bottom {
		bottom: 30px;
		left: 50%;
		transform: translateX(-50%);
		width: 90%;
		max-width: 700px;
	}
	.panel-bottom.minimized {
		transform: translate(-50%, calc(100% + 24px));
	}

	/* Panel header with toggle */
	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.75rem 0.5rem 1rem;
		border-bottom: 1px solid hsl(var(--border) / 0.3);
		background: rgba(0, 0, 0, 0.15);
		flex-shrink: 0;
	}
	.panel-title {
		font-size: 0.75rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: hsl(var(--muted-foreground));
	}
	.panel-toggle {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 1.75rem;
		height: 1.75rem;
		border: none;
		border-radius: 0.375rem;
		background: transparent;
		color: hsl(var(--muted-foreground));
		cursor: pointer;
		transition: color 0.2s, background 0.2s;
	}
	.panel-toggle:hover {
		color: hsl(var(--foreground));
		background: hsl(var(--border) / 0.3);
	}
	.panel-content {
		overflow-y: auto;
	}

	/* Expand button when panel is minimized */
	.panel-expand-btn {
		position: fixed;
		z-index: 11;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 2rem;
		height: 2.5rem;
		border: 1px solid hsl(var(--border) / 0.5);
		border-radius: 0 0.375rem 0.375rem 0;
		background: rgba(0, 0, 0, 0.5);
		backdrop-filter: blur(8px);
		-webkit-backdrop-filter: blur(8px);
		color: hsl(var(--muted-foreground));
		cursor: pointer;
		transition: color 0.2s, background 0.2s, opacity 0.2s;
	}
	.panel-expand-btn:hover {
		color: hsl(var(--foreground));
		background: rgba(0, 0, 0, 0.65);
	}
	.panel-expand-left {
		top: 50%;
		left: 0;
		transform: translateY(-50%);
		border-radius: 0 0.375rem 0.375rem 0;
	}
	.panel-expand-right {
		top: 50%;
		right: 0;
		transform: translateY(-50%);
		border-radius: 0.375rem 0 0 0.375rem;
		border-left: none;
	}
	.panel-expand-bottom {
		bottom: 0;
		left: 50%;
		transform: translateX(-50%);
		width: 3rem;
		height: 2rem;
		border-radius: 0.375rem 0.375rem 0 0;
		border-bottom: none;
	}

	/* Floating side panels — inner content */
	.floating-panel,
	.sidebar-container {
		max-width: 350px;
		max-height: calc(100vh - 180px);
		overflow-y: auto;
		background: transparent;
		border: none;
		border-radius: 0;
		padding: 1rem;
		color: hsl(var(--foreground));
	}
	.floating-panel-left,
	.floating-panel-right {
		position: static;
	}
	.floating-panel-section {
		padding: 0.75rem;
		border-radius: 0.5rem;
		background: rgba(0, 0, 0, 0.25);
		border: 1px solid hsl(var(--border) / 0.3);
		margin-bottom: 0.75rem;
		color: hsl(var(--foreground));
	}

	/* Floating bottom controls — inner */
	.floating-bottom-controls {
		position: static;
		bottom: auto;
		left: auto;
		transform: none;
		width: 100%;
		max-width: none;
		background: transparent;
		backdrop-filter: none;
		border: none;
		border-radius: 0;
		padding: 1rem;
		color: hsl(var(--foreground));
	}
</style>
