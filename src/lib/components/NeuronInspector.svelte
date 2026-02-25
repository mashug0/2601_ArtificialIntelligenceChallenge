<script lang="ts">
	import { activeNode, selectionStore } from '$lib/stores/selection';
	import { COLORS } from '$lib/utils/colorScales';

	$: node = $activeNode;
</script>

<div class="neuron-inspector panel">
	<div class="panel-header">Neuron Inspector</div>

	<div class="panel-body">
		{#if node}
			<div class="space-y-4 fade-enter">
				<!-- Node ID -->
				<div class="flex items-center justify-between">
					<span class="text-paper-muted">ID</span>
					<span class="font-mono text-sm">{node.id}</span>
				</div>

				<!-- Layer -->
				<div class="flex items-center justify-between">
					<span class="text-paper-muted">Layer</span>
					<span class="font-mono">{node.layer + 1}</span>
				</div>

				<!-- Neuron Index -->
				<div class="flex items-center justify-between">
					<span class="text-paper-muted">Index</span>
					<span class="font-mono">{node.neuron_idx}</span>
				</div>

				<!-- Divider -->
				<hr class="border-gray-100" />

				<!-- Excitatory Activation -->
				<div>
					<div class="flex items-center justify-between mb-1">
						<span class="text-paper-muted flex items-center gap-2">
							<span class="w-3 h-3 rounded-full" style="background-color: {COLORS.excitatory}"></span>
							Excitatory (x_N)
						</span>
						<span class="font-mono text-sm">{node.excitatory_activation.toFixed(4)}</span>
					</div>
					<div class="h-2 rounded overflow-hidden" style="background: rgba(0, 0, 0, 0.35);">
						<div
							class="h-full transition-all duration-300"
							style="width: {Math.min(node.excitatory_activation / 2, 1) * 100}%; background-color: {COLORS.excitatory}"
						></div>
					</div>
				</div>

				<!-- Inhibitory Activation -->
				<div>
					<div class="flex items-center justify-between mb-1">
						<span class="text-paper-muted flex items-center gap-2">
							<span class="w-3 h-3 rounded-full" style="background-color: {COLORS.inhibitory}"></span>
							Inhibitory (y)
						</span>
						<span class="font-mono text-sm">{node.inhibitory_activation.toFixed(4)}</span>
					</div>
					<div class="h-2 rounded overflow-hidden" style="background: rgba(0, 0, 0, 0.35);">
						<div
							class="h-full transition-all duration-300"
							style="width: {Math.min(node.inhibitory_activation / 2, 1) * 100}%; background-color: {COLORS.inhibitory}"
						></div>
					</div>
				</div>

				<!-- Signal Type -->
				<div class="flex items-center justify-between">
					<span class="text-paper-muted">Signal Type</span>
					<span
						class="px-2 py-0.5 rounded text-xs font-medium capitalize"
						class:bg-excitatory={node.signal_type === 'excitatory'}
						class:bg-inhibitory={node.signal_type === 'inhibitory'}
						class:text-white={node.signal_type === 'excitatory' || node.signal_type === 'inhibitory'}
						class:bg-secondary={node.signal_type === 'inactive'}
						style={node.signal_type === 'inactive' ? 'background: rgba(0, 0, 0, 0.35);' : ''}
					>
						{node.signal_type}
					</span>
				</div>

				<!-- Specialist indicator -->
				{#if node.is_specialist}
					<div class="p-3 bg-specialist/10 border border-specialist/20 rounded">
						<div class="flex items-center gap-2 text-specialist font-medium">
							<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
								<path
									d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
								/>
							</svg>
							<span>Specialist Neuron</span>
						</div>
						<p class="mt-2 text-sm text-specialist/80">
							Primary trigger: <code class="px-1 py-0.5 bg-specialist/20 rounded font-mono">
								'{node.specialist_char === ' ' ? 'space' : node.specialist_char}'
							</code>
						</p>
					</div>
				{/if}
			</div>
		{:else}
			<div class="text-center py-8 text-paper-muted">
				<svg class="w-12 h-12 mx-auto mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="1"
						d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
					/>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="1"
						d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
					/>
				</svg>
				<p class="text-sm">Hover or click on a neuron to inspect its properties</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.fade-enter {
		animation: fadeIn 0.2s ease-out;
	}

	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateY(-4px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
</style>
