<script lang="ts">
	import { COLORS } from '$lib/utils/colorScales';
</script>

<div class="legend panel">
	<div class="panel-header">Legend</div>

	<div class="panel-body space-y-4">
		<!-- Signal types -->
		<div>
			<h4 class="text-xs font-medium text-paper-muted uppercase tracking-wider mb-2">Signal Type</h4>
			<div class="space-y-2">
				<div class="flex items-center gap-3">
					<div class="w-4 h-4 rounded-full" style="background-color: {COLORS.excitatory}"></div>
					<span class="text-sm">Excitatory (x_N)</span>
				</div>
				<div class="flex items-center gap-3">
					<div class="w-4 h-4 rounded-full" style="background-color: {COLORS.inhibitory}"></div>
					<span class="text-sm">Inhibitory (y)</span>
				</div>
				<div class="flex items-center gap-3">
					<div class="w-4 h-4 rounded-full" style="background-color: {COLORS.inactive}"></div>
					<span class="text-sm">Inactive</span>
				</div>
			</div>
		</div>

		<!-- Specialist -->
		<div>
			<h4 class="text-xs font-medium text-paper-muted uppercase tracking-wider mb-2">Special</h4>
			<div class="flex items-center gap-3">
				<div class="relative">
					<div class="w-4 h-4 rounded-full neuron-specialist" style="background-color: {COLORS.specialist}"></div>
				</div>
				<span class="text-sm">Specialist Neuron</span>
			</div>
		</div>

		<!-- Size encoding -->
		<div>
			<h4 class="text-xs font-medium text-paper-muted uppercase tracking-wider mb-2">Node Size</h4>
			<div class="flex items-center gap-4">
				<div class="flex items-center gap-2">
					<div class="w-2 h-2 rounded-full" style="background: rgba(0, 0, 0, 0.35);"></div>
					<span class="text-xs text-paper-muted">Low</span>
				</div>
				<div class="flex-1 h-1 bg-gradient-to-r from-secondary to-excitatory rounded"></div>
				<div class="flex items-center gap-2">
					<div class="w-4 h-4 rounded-full bg-excitatory"></div>
					<span class="text-xs text-paper-muted">High</span>
				</div>
			</div>
			<p class="text-xs text-paper-muted mt-1">Size indicates activation strength</p>
		</div>

		<!-- Connections -->
		<div>
			<h4 class="text-xs font-medium text-paper-muted uppercase tracking-wider mb-2">Connections</h4>
			<div class="space-y-2">
				<div class="flex items-center gap-3">
					<div class="w-8 h-0.5 rounded" style="background: rgba(0, 0, 0, 0.35);"></div>
					<span class="text-sm">Inter-layer flow</span>
				</div>
			</div>
			<p class="text-xs text-paper-muted mt-1">Opacity indicates connection strength</p>
		</div>

		<!-- Keyboard shortcuts -->
		<div class="pt-3 border-t border-gray-100">
			<h4 class="text-xs font-medium text-paper-muted uppercase tracking-wider mb-2">Keyboard</h4>
			<div class="grid grid-cols-2 gap-2 text-xs">
				<div class="flex items-center gap-2">
					<kbd class="px-1.5 py-0.5 rounded font-mono text-muted-foreground" style="background: rgba(0, 0, 0, 0.35);">Space</kbd>
					<span class="text-paper-muted">Play/Pause</span>
				</div>
				<div class="flex items-center gap-2">
					<kbd class="px-1.5 py-0.5 rounded font-mono text-muted-foreground" style="background: rgba(0, 0, 0, 0.35);">&larr; &rarr;</kbd>
					<span class="text-paper-muted">Step</span>
				</div>
				<div class="flex items-center gap-2">
					<kbd class="px-1.5 py-0.5 rounded font-mono text-muted-foreground" style="background: rgba(0, 0, 0, 0.35);">Home</kbd>
					<span class="text-paper-muted">Reset</span>
				</div>
			</div>
		</div>
	</div>
</div>
