<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { playbackStore } from '$lib/stores/playback';
	import type { Vocabulary, SpecialistNeuron } from '$lib/types';

	export let tokens: number[];
	export let vocabulary: Vocabulary;
	export let specialists: SpecialistNeuron[] = [];

	let animationFrame: number | null = null;
	let lastTickTime = 0;

	// Build set of characters that have specialist neurons
	$: specialistChars = new Set(specialists.map((s) => s.trigger_char));

	// Get character for token
	function getChar(tokenIdx: number): string {
		const char = vocabulary.int_to_char[String(tokenIdx)];
		if (char === '\n') return '\\n';
		if (char === ' ') return '\u2423'; // visible space
		if (char === '\t') return '\\t';
		return char || '?';
	}

	// Check if token is a specialist trigger
	function isSpecialistToken(tokenIdx: number): boolean {
		const char = vocabulary.int_to_char[String(tokenIdx)];
		return specialistChars.has(char);
	}

	// Animation loop
	function startAnimation() {
		const tokenDuration = 1000 / $playbackStore.speed;
		lastTickTime = performance.now();

		function tick(currentTime: number) {
			if (!$playbackStore.isPlaying) return;

			const elapsed = currentTime - lastTickTime;
			if (elapsed >= tokenDuration) {
				playbackStore.nextToken();
				lastTickTime = currentTime;
			}

			animationFrame = requestAnimationFrame(tick);
		}

		animationFrame = requestAnimationFrame(tick);
	}

	function stopAnimation() {
		if (animationFrame !== null) {
			cancelAnimationFrame(animationFrame);
			animationFrame = null;
		}
	}

	// React to play state changes
	$: if ($playbackStore.isPlaying) {
		startAnimation();
	} else {
		stopAnimation();
	}

	// Keyboard shortcuts
	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

		switch (e.key) {
			case ' ':
				e.preventDefault();
				playbackStore.togglePlay();
				break;
			case 'ArrowRight':
				e.preventDefault();
				playbackStore.nextToken();
				break;
			case 'ArrowLeft':
				e.preventDefault();
				playbackStore.prevToken();
				break;
			case 'Home':
				e.preventDefault();
				playbackStore.reset();
				break;
		}
	}

	onMount(() => {
		playbackStore.setMaxToken(tokens.length - 1);
		window.addEventListener('keydown', handleKeydown);
	});

	onDestroy(() => {
		stopAnimation();
		window.removeEventListener('keydown', handleKeydown);
	});
</script>

<div class="token-timeline border-t border-border/70" style="background: rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);">
	<!-- Token sequence -->
	<div class="tokens-container px-4 pt-4 pb-2">
		<div class="flex flex-wrap gap-1 max-h-24 overflow-y-auto scrollbar-thin">
			{#each tokens as token, i}
				<button
					class="token-chip"
					class:active={i === $playbackStore.currentToken}
					class:specialist={isSpecialistToken(token)}
					on:click={() => playbackStore.setToken(i)}
					title="Token {i}: '{vocabulary.int_to_char[String(token)]}'"
				>
					{getChar(token)}
				</button>
			{/each}
		</div>
	</div>

	<!-- Controls -->
	<div class="controls px-4 py-3 flex items-center gap-4 border-t border-border/60">
		<!-- Play/Pause -->
		<button
			class="play-btn w-10 h-10 flex items-center justify-center bg-indigo-600 hover:bg-indigo-700 text-white rounded-full transition-colors"
			on:click={() => playbackStore.togglePlay()}
			title={$playbackStore.isPlaying ? 'Pause (Space)' : 'Play (Space)'}
		>
			{#if $playbackStore.isPlaying}
				<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
					<path
						fill-rule="evenodd"
						d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"
						clip-rule="evenodd"
					/>
				</svg>
			{:else}
				<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
					<path
						fill-rule="evenodd"
						d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
						clip-rule="evenodd"
					/>
				</svg>
			{/if}
		</button>

		<!-- Step buttons -->
		<div class="flex gap-1">
			<button
				class="p-2 rounded transition-colors hover:opacity-100"
				style="background: rgba(0, 0, 0, 0.25);"
				on:click={() => playbackStore.prevToken()}
				title="Previous token (Left arrow)"
			>
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
				</svg>
			</button>
			<button
				class="p-2 rounded transition-colors hover:opacity-100"
				style="background: rgba(0, 0, 0, 0.25);"
				on:click={() => playbackStore.nextToken()}
				title="Next token (Right arrow)"
			>
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
				</svg>
			</button>
		</div>

		<!-- Scrubber -->
		<div class="flex-1">
			<input
				type="range"
				min="0"
				max={tokens.length - 1}
				bind:value={$playbackStore.currentToken}
				on:input={(e) => playbackStore.setToken(parseInt(e.currentTarget.value))}
				class="w-full h-2 rounded-lg appearance-none cursor-pointer accent-indigo-400"
				style="background: rgba(0, 0, 0, 0.35);"
			/>
		</div>

		<!-- Speed control -->
		<select
			bind:value={$playbackStore.speed}
			on:change={(e) => playbackStore.setSpeed(parseFloat(e.currentTarget.value))}
			class="px-2 py-1 border border-border rounded text-sm font-mono"
			style="background: rgba(0, 0, 0, 0.35);"
		>
			<option value={0.25}>0.25x</option>
			<option value={0.5}>0.5x</option>
			<option value={1}>1x</option>
			<option value={2}>2x</option>
			<option value={4}>4x</option>
		</select>

		<!-- Loop toggle -->
		<button
			class="p-2 rounded transition-colors"
			class:bg-indigo-900={$playbackStore.loop}
			class:text-indigo-300={$playbackStore.loop}
			style={$playbackStore.loop ? 'background-color: rgb(30 27 75 / 0.6);' : 'background: rgba(0, 0, 0, 0.25);'}
			on:click={() => playbackStore.toggleLoop()}
			title="Toggle loop"
		>
			<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
				/>
			</svg>
		</button>

		<!-- Progress indicator -->
		<span class="font-mono text-sm text-paper-muted min-w-[80px] text-right">
			{$playbackStore.currentToken + 1} / {tokens.length}
		</span>
	</div>
</div>

<style>
	.token-chip {
		@apply px-2 py-1 rounded text-sm font-mono transition-all duration-200 text-muted-foreground;
		background: rgba(0, 0, 0, 0.35);
	}
	.token-chip:hover {
		background: rgba(0, 0, 0, 0.5);
	}

	.token-chip.active {
		@apply bg-excitatory text-white shadow-md scale-110;
	}

	.token-chip.specialist {
		@apply ring-2 ring-specialist ring-offset-1;
	}

	input[type='range']::-webkit-slider-thumb {
		@apply appearance-none w-4 h-4 rounded-full bg-indigo-600 cursor-pointer;
	}

	input[type='range']::-moz-range-thumb {
		@apply w-4 h-4 rounded-full bg-indigo-600 cursor-pointer border-0;
	}
</style>
