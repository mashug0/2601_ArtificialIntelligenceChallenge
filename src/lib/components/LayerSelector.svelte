<script lang="ts">
	import { layerStore } from '$lib/stores/selection';
	import { playbackStore } from '$lib/stores/playback';
	import type { ActivationData } from '$lib/types';
	import { getLayerStats } from '$lib/utils/graphBuilder';

	export let data: ActivationData;
	export let sequenceIdx: number = 0;

	// Get layer stats for current token
	$: layerStats = getLayerStats(data, $playbackStore.currentToken, sequenceIdx);

	// Count of expanded layers
	$: expandedCount = $layerStore.expanded.filter(Boolean).length;
</script>

<div class="layer-selector panel">
	<div class="panel-header flex items-center justify-between">
		<span>Layers</span>
		<span class="text-sm font-mono text-paper-muted">{expandedCount}/8 visible</span>
	</div>

	<div class="panel-body">
		<!-- Bulk actions -->
		<div class="flex gap-2 mb-3 pb-3 border-b border-border/60">
			<button
				class="flex-1 px-2 py-1 text-xs rounded transition-colors hover:opacity-90"
				style="background: rgba(0, 0, 0, 0.35);"
				on:click={() => layerStore.expandAll()}
			>
				Expand All
			</button>
			<button
				class="flex-1 px-2 py-1 text-xs rounded transition-colors hover:opacity-90"
				style="background: rgba(0, 0, 0, 0.35);"
				on:click={() => layerStore.collapseAll()}
			>
				Collapse All
			</button>
		</div>

		<!-- Layer list -->
		<div class="space-y-1">
			{#each layerStats as stats, i}
				<button
					class="layer-toggle w-full"
					class:collapsed={!$layerStore.expanded[i]}
					on:click={() => layerStore.toggleExpanded(i)}
				>
					<!-- Expand/collapse icon -->
					<svg
						class="w-4 h-4 transition-transform"
						class:rotate-90={$layerStore.expanded[i]}
						fill="none"
						stroke="currentColor"
						viewBox="0 0 24 24"
					>
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
					</svg>

					<!-- Layer label -->
					<span class="flex-1 text-left font-mono text-sm">
						Layer {i + 1}
					</span>

					<!-- Activity indicators -->
					<div class="flex items-center gap-2">
						<!-- Excitatory count -->
						<div class="flex items-center gap-1" title="Excitatory neurons">
							<div class="w-2 h-2 rounded-full bg-excitatory"></div>
							<span class="text-xs font-mono text-paper-muted">{stats.excitatoryCount}</span>
						</div>

						<!-- Inhibitory count -->
						<div class="flex items-center gap-1" title="Inhibitory neurons">
							<div class="w-2 h-2 rounded-full bg-inhibitory"></div>
							<span class="text-xs font-mono text-paper-muted">{stats.inhibitoryCount}</span>
						</div>

						<!-- Sparsity -->
						<span class="text-xs font-mono text-paper-muted min-w-[40px] text-right">
							{(stats.sparsity * 100).toFixed(1)}%
						</span>
					</div>
				</button>
			{/each}
		</div>

		<!-- Legend -->
		<div class="mt-4 pt-3 border-t border-border/60">
			<p class="text-xs text-paper-muted">
				Click to expand/collapse layers in the 3D view. Percentages show firing rate at current token.
			</p>
		</div>
	</div>
</div>

<style>
	.layer-toggle {
		@apply flex items-center gap-2 px-3 py-2 rounded-md transition-colors;
	}
	.layer-toggle:hover {
		background: rgba(0, 0, 0, 0.25);
	}

	.layer-toggle.collapsed {
		@apply opacity-50;
	}

	.layer-toggle.collapsed:hover {
		@apply opacity-75;
	}
</style>
