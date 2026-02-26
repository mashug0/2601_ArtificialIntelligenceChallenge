<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { Vocabulary, ActivationData } from '$lib/types';

	export let vocabulary: Vocabulary;
	export let maxLength: number = 128;
	export let isLoading: boolean = false;
	export let apiUrl: string = '';

	const dispatch = createEventDispatcher<{
		submit: { activationData: ActivationData };
		error: { message: string };
	}>();

	let inputText = '';
	let error = '';
	let apiError = '';

	// BPE tokenizer accepts any text - no char-level validation needed
	function handleInput(e: Event) {
		const target = e.target as HTMLTextAreaElement;
		inputText = target.value.slice(0, maxLength);
		apiError = '';
		error = '';
	}

	// Call backend API for inference
	async function runInference(text: string): Promise<ActivationData> {
		const response = await fetch(`${apiUrl}/api/infer`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ text, max_length: maxLength })
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(errorData.detail || `API error: ${response.status}`);
		}

		const data = await response.json();

		// Fetch vocabulary from API to get proper BPE mappings
		let apiVocab = vocabulary;
		try {
			const vocabRes = await fetch(`${apiUrl}/vocabulary`);
			if (vocabRes.ok) {
				apiVocab = await vocabRes.json();
			}
		} catch {}

		// Convert API response to ActivationData format
		return {
			metadata: data.metadata,
			vocabulary: apiVocab,
			specialists: [],
			sequences: [
				{
					id: 'custom_input',
					input_text: data.input_text,
					tokens: data.tokens,
					layer_activations: data.layer_activations.map((layer: any) => ({
						layer_idx: layer.layer_idx,
						sparsity: layer.sparsity,
						sampled_indices: layer.sampled_indices,
						excitatory: layer.excitatory,
						inhibitory: layer.inhibitory
					}))
				}
			]
		};
	}

	// Handle submit
	async function handleSubmit() {
		if (error || inputText.length === 0 || isLoading) return;

		isLoading = true;
		apiError = '';

		try {
			const activationData = await runInference(inputText);
			dispatch('submit', { activationData });
		} catch (e) {
			const message = e instanceof Error ? e.message : 'Unknown error';
			apiError = message;
			dispatch('error', { message });
		} finally {
			isLoading = false;
		}
	}

	// Preset examples
	const examples = [
		{ label: 'War & Peace', text: 'The general ordered the soldiers to march across the river.' },
		{ label: 'Sherlock', text: 'He examined the curious marks upon the door with his glass.' },
		{ label: 'Nature', text: 'The sun rose slowly over the dark mountain and lit the valley.' },
		{ label: 'Battle', text: 'The army advanced through the snow while the enemy watched.' }
	];

	function loadExample(text: string) {
		inputText = text.slice(0, maxLength);
		error = '';
		apiError = '';
	}
</script>

<div class="custom-input panel">
	<div class="panel-header flex items-center justify-between">
		<span>Custom Input</span>
		<span class="text-sm font-mono text-paper-muted">
			{inputText.length}/{maxLength}
		</span>
	</div>

	<div class="panel-body space-y-3">
		<!-- Text input -->
	<textarea
			bind:value={inputText}
			on:input={handleInput}
			placeholder="Enter text to visualize (BPE tokenizer — any text works)"
			class="w-full h-24 px-3 py-2 border border-border rounded font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-foreground"
			style="background: rgba(0, 0, 0, 0.35);"
			class:border-red-300={!!apiError}
			disabled={isLoading}
		/>

		<!-- Error messages -->
		{#if apiError}
			<div class="p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
				<strong>API Error:</strong> {apiError}
				<p class="text-xs mt-1 text-red-500">
					Make sure the backend is running: <code class="bg-red-100 px-1 rounded">cd backend && python main.py</code>
				</p>
			</div>
		{/if}

		<!-- Example presets -->
		<div class="flex flex-wrap gap-2">
			<span class="text-sm text-paper-muted">Examples:</span>
			{#each examples as example}
				<button
					class="px-2 py-1 text-xs rounded transition-colors hover:opacity-90"
					style="background: rgba(0, 0, 0, 0.35);"
					on:click={() => loadExample(example.text)}
					disabled={isLoading}
				>
					{example.label}
				</button>
			{/each}
		</div>

		<!-- Submit button -->
		<button
			class="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-secondary/60 disabled:cursor-not-allowed text-white rounded transition-colors flex items-center justify-center gap-2"
			on:click={handleSubmit}
			disabled={inputText.length === 0 || isLoading}
		>
			{#if isLoading}
				<div class="spinner"></div>
				<span>Processing...</span>
			{:else}
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M13 10V3L4 14h7v7l9-11h-7z"
					/>
				</svg>
				<span>Visualize</span>
			{/if}
		</button>

		<!-- API info -->
		<p class="text-xs text-paper-muted">
			Backend: <code class="px-1 rounded font-mono" style="background: rgba(0, 0, 0, 0.35);">{apiUrl}</code>
		</p>
	</div>
</div>

<style>
	.spinner {
		@apply w-4 h-4 border-2 border-white/30 rounded-full;
		border-top-color: white;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
