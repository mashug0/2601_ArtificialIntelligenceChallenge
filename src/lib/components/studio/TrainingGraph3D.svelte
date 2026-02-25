<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  export let snapshot: { nodes: any[]; links: any[] } | null = null;
  export let epoch: number = 0;
  export let batch: number = 0;
  export let loss: number = 0;
  export let valAcc: number = 0;
  export let nLayers: number = 4;
  // inferMode: when true, nodes are colored/sized by activation strength instead of layer
  export let inferMode: boolean = false;

  let container: HTMLDivElement;
  let Graph: any = null;

  // Layer color palette — one bright color per layer
  const LAYER_COLORS = [
    '#f59e0b', // amber   — layer 0
    '#34d399', // emerald — layer 1
    '#60a5fa', // blue    — layer 2
    '#f87171', // red     — layer 3
    '#a78bfa', // violet  — layer 4
    '#22d3ee', // cyan    — layer 5
    '#fb923c', // orange  — layer 6
    '#e879f9', // fuchsia — layer 7
  ];

  // ── Link type filter toggles ─────────────────────────────────────
  let showAttention = true;
  let showResidual  = true;

  // ── Warm in-place graph update ──────────────────────────────────
  let liveNodes: Map<string, any> = new Map();
  let liveLinks: Map<string, any> = new Map();

  function updateGraph(s: { nodes: any[]; links: any[] }) {
    if (!Graph) return;

    const nextNodes: any[] = s.nodes.map((n: any) => {
      if (liveNodes.has(n.id)) {
        const existing = liveNodes.get(n.id);
        existing.enc_norm  = n.enc_norm;
        existing.dec_norm  = n.dec_norm;
        existing.ln_weight = n.ln_weight;
        existing.layer     = n.layer;
        existing.head      = n.head;
        existing.activation = n.activation;  // ← Fix: update activation for inference mode
        return existing;
      }
      const fresh = { ...n };
      liveNodes.set(n.id, fresh);
      return fresh;
    });
    liveNodes = new Map(nextNodes.map((n: any) => [n.id, n]));

    const nextLinks: any[] = s.links.map((l: any) => {
      const srcId = typeof l.source === 'object' ? l.source.id : l.source;
      const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
      const key = `${srcId}→${tgtId}`;
      if (liveLinks.has(key)) {
        liveLinks.get(key).weight = l.weight;
        liveLinks.get(key).rtype  = l.rtype;
        return liveLinks.get(key);
      }
      const newLink = { source: srcId, target: tgtId, weight: l.weight, rtype: l.rtype };
      liveLinks.set(key, newLink);
      return newLink;
    });
    liveLinks = new Map(nextLinks.map((l: any) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return [`${s}→${t}`, l];
    }));

    Graph.graphData({ nodes: nextNodes, links: nextLinks });
    setupLayerForces(); // re-apply after every graphData() — library wipes custom forces on each call
  }

  // Re-run when snapshot changes OR when link-type filters are toggled
  $: if (Graph && snapshot) {
    // Reference filter vars so Svelte tracks them as reactive dependencies
    const _sa = showAttention; const _sr = showResidual;
    updateGraph({
      nodes: snapshot.nodes,
      links: snapshot.links.filter((l: any) => l.rtype === 'attention' ? _sa : _sr),
    });
  }
  // Refresh node appearance when inferMode toggles (colors/sizes use inferMode at call time)
  $: if (Graph) Graph.nodeColor(Graph.nodeColor()).nodeVal(Graph.nodeVal());

  function setupLayerForces() {
    if (!Graph) return;
    const LAYER_SPACING_Z = 130;
    const LAYER_SPREAD_X  = 320;
    const CLUSTER_RADIUS  = 120;

    const layerForceZ = (alpha: number) => {
      Graph.graphData().nodes.forEach((node: any) => {
        const targetZ = node.layer * LAYER_SPACING_Z - ((nLayers - 1) * LAYER_SPACING_Z) / 2;
        node.vz = (node.vz || 0) + (targetZ - (node.z || 0)) * alpha * 0.2;
      });
    };

    const layerClusterXY = (alpha: number) => {
      Graph.graphData().nodes.forEach((node: any) => {
        const cx = node.layer * LAYER_SPREAD_X - ((nLayers - 1) * LAYER_SPREAD_X) / 2;
        const dx = (node.x || 0) - cx;
        const dy = (node.y || 0);
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > CLUSTER_RADIUS) {
          const f = (dist - CLUSTER_RADIUS) * 0.55 * alpha;
          node.vx = (node.vx || 0) - (dx / dist) * f;
          node.vy = (node.vy || 0) - (dy / dist) * f;
        }
      });
    };

    const intraLayerRepulsion = (alpha: number) => {
      const byLayer: Map<number, any[]> = new Map();
      Graph.graphData().nodes.forEach((n: any) => {
        if (!byLayer.has(n.layer)) byLayer.set(n.layer, []);
        byLayer.get(n.layer)!.push(n);
      });
      byLayer.forEach(layerNodes => {
        for (let i = 0; i < layerNodes.length; i++) {
          for (let j = i + 1; j < layerNodes.length; j++) {
            const dx = (layerNodes[j].x || 0) - (layerNodes[i].x || 0);
            const dy = (layerNodes[j].y || 0) - (layerNodes[i].y || 0);
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            if (dist < 50) {
              const f = (-30 * alpha) / (dist * dist);
              layerNodes[i].vx = (layerNodes[i].vx || 0) - (dx / dist) * f;
              layerNodes[i].vy = (layerNodes[i].vy || 0) - (dy / dist) * f;
              layerNodes[j].vx = (layerNodes[j].vx || 0) + (dx / dist) * f;
              layerNodes[j].vy = (layerNodes[j].vy || 0) + (dy / dist) * f;
            }
          }
        }
      });
    };

    Graph.d3Force('layerZ',     layerForceZ);
    Graph.d3Force('layerXY',    layerClusterXY);
    Graph.d3Force('intraLayer', intraLayerRepulsion);

    // Link force must stay active for D3 to resolve string IDs → node objects
    const linkForce = Graph.d3Force('link');
    if (linkForce) {
      linkForce
        .strength((l: any) => (l.rtype === 'residual' ? 0.015 : 0.008))
        .distance((l: any) => (l.rtype === 'residual' ? LAYER_SPREAD_X : 50));
    }
  }

  onMount(async () => {
    const mod = await import('3d-force-graph');
    const ForceGraph3D = mod.default;

    Graph = ForceGraph3D({ rendererConfig: { alpha: true, antialias: true } })(container)
      .backgroundColor('rgba(0,0,0,0)')
      .showNavInfo(false)
      // ── Nodes ──────────────────────────────────────────────────
      .nodeColor((n: any) => {
        if (inferMode && n.activation != null) {
          // Heat map: dim layer color → bright white as activation → 1
          const a = Math.min(Math.max(n.activation ?? 0, 0), 1);
          const base = LAYER_COLORS[n.layer ?? 0] ?? '#94a3b8';
          if (a < 0.05) return '#1e293b';  // nearly silent → dark slate
          // Blend: low activation = amber, high = white-hot
          const v = Math.round(100 + a * 155);
          const hex = v.toString(16).padStart(2, '0');
          return `#ff${hex}${Math.round(a * 80).toString(16).padStart(2,'0')}`;  // red-orange-white heat
        }
        return LAYER_COLORS[n.layer ?? 0] ?? '#94a3b8';
      })
      .nodeVal((n: any) => {
        if (inferMode && n.activation != null) {
          // Fired neurons grow larger
          return 4 + (n.activation ?? 0) * 20;
        }
        return 8;
      })
      .nodeOpacity(0.95)
      .nodeResolution(12)
      .nodeLabel((n: any) => {
        const act = n.activation != null ? ` | act: ${(n.activation ?? 0).toFixed(3)}` : '';
        return `Layer ${n.layer} · Head ${n.head} · N${n.neuron} | enc: ${(n.enc_norm ?? 0).toFixed(3)} dec: ${(n.dec_norm ?? 0).toFixed(3)}${act}`;
      })
      // ── Links ──────────────────────────────────────────────────
      // linkOpacity only accepts a static number (not a function) — passing a function silently does nothing
      .linkOpacity(0.85)
      // linkColor encodes weight strength as brightness (linkColor DOES accept functions)
      .linkColor((l: any) => {
        const w = Math.abs(l.weight ?? 0);
        const h = (n: number) => Math.round(Math.min(Math.max(n, 0), 255)).toString(16).padStart(2, '0');
        if (l.rtype === 'residual') {
          // residual: sky-blue, brighter = stronger weight
          const v = 70 + w * 700;
          return `#${h(v * 0.5)}${h(v * 0.8)}${h(v)}`;
        }
        // attention: amber/gold, brighter = stronger weight
        const v = 80 + w * 700;
        return `#${h(v)}${h(v * 0.65)}${h(10)}`;
      })
      // linkWidth as function — weight-based thickness, floor ensures always visible
      .linkWidth((l: any) => {
        const w = Math.abs(l.weight ?? 0);
        if (l.rtype === 'residual') return w * 2 + 0.8;
        return w * 4 + 1.2;
      })
      // ── Particles ──────────────────────────────────────────────
      .linkDirectionalParticles((l: any) => {
        if (l.rtype === 'attention') return 3;
        return Math.abs(l.weight ?? 0) > 0.2 ? 2 : 0;
      })
      .linkDirectionalParticleSpeed(0.005)
      .linkDirectionalParticleWidth(2.5)
      .linkDirectionalParticleColor((l: any) => {
        if (l.rtype === 'residual') return '#7dd3fc';  // sky blue for residual
        // attention: source layer color
        const srcId = typeof l.source === 'object' ? l.source.id : l.source;
        const srcNode = Graph.graphData().nodes.find((n: any) => n.id === srcId);
        if (srcNode) return LAYER_COLORS[srcNode.layer ?? 0] ?? '#f59e0b';
        return '#f59e0b';
      })
      .width(container.clientWidth || 800)
      .height(container.clientHeight || 300);

    Graph.graphData({ nodes: [], links: [] });
    setupLayerForces();

    // Make the WebGL canvas truly transparent so the glass panel shows through
    const renderer = Graph.renderer();
    if (renderer) {
      renderer.setClearColor(0x000000, 0);  // black with 0 alpha = fully transparent
      renderer.setPixelRatio(window.devicePixelRatio);
    }

    const ro = new ResizeObserver(() => {
      if (Graph) {
        Graph.width(container.clientWidth);
        Graph.height(container.clientHeight);
      }
    });
    ro.observe(container);

    return () => ro.disconnect();
  });

  onDestroy(() => {
    if (Graph) {
      try { Graph._destructor?.(); } catch {}
      Graph = null;
    }
  });

  $: visibleLayers = snapshot
    ? [...new Set(snapshot.nodes.map((n: any) => n.layer ?? 0))].sort((a, b) => a - b)
    : [];

  $: filteredLinkCount = snapshot
    ? snapshot.links.filter((l: any) => l.rtype === 'attention' ? showAttention : showResidual).length
    : 0;
</script>

<div class="relative w-full h-full studio-graph-root">
  <!-- Glass layer behind canvas so panel matches side panels when WebGL is transparent -->
  <div class="studio-graph-glass-bg" aria-hidden="true"/>

  <div bind:this={container} class="relative w-full h-full z-[1]"/>

  {#if epoch > 0 || loss > 0}
    <div class="absolute top-3 left-3 font-mono text-xs space-y-0.5 pointer-events-none z-10">
      <div class="glass-badge text-amber-400 px-2.5 py-1 rounded-md border border-amber-500/40">
        epoch <span class="font-bold">{epoch}</span>
        &nbsp;·&nbsp; loss <span class="font-bold">{loss.toFixed(4)}</span>
        {#if valAcc > 0}&nbsp;·&nbsp; val <span class="text-emerald-400 font-bold">{valAcc.toFixed(1)}%</span>{/if}
      </div>
      {#if batch > 0}
        <div class="glass-badge-muted text-muted-foreground px-2 py-0.5 rounded text-[10px]">
          batch {batch}
        </div>
      {/if}
    </div>
  {/if}

  {#if snapshot && snapshot.nodes.length > 0}
    <div class="absolute bottom-3 left-3 font-mono text-[10px] pointer-events-none z-10">
      <div class="glass-badge px-2.5 py-2 rounded-md space-y-1 border border-border/50">
        <div class="text-muted-foreground text-[9px] uppercase tracking-wider mb-1">
          {inferMode ? '🔥 inference — neuron activations' : 'layers (color + position)'}
        </div>
        {#if inferMode}
          <div class="text-[9px] mb-1.5 space-y-0.5">
            <div class="flex items-center gap-1.5"><div class="w-2.5 h-2.5 rounded-full bg-white flex-shrink-0"/><span class="text-foreground/80">high activation</span></div>
            <div class="flex items-center gap-1.5"><div class="w-2.5 h-2.5 rounded-full bg-orange-500 flex-shrink-0"/><span class="text-foreground/80">medium activation</span></div>
            <div class="flex items-center gap-1.5"><div class="w-2.5 h-2.5 rounded-full bg-slate-600 flex-shrink-0"/><span class="text-foreground/80">silent neuron</span></div>
          </div>
        {/if}
        <div class="text-muted-foreground text-[9px] mb-1.5">{visibleLayers.length} layers · {snapshot.nodes.length} neurons · {filteredLinkCount} synapses</div>
        {#each visibleLayers.slice(0, 8) as l}
          <div class="flex items-center gap-1.5">
            <div class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background:{LAYER_COLORS[l] ?? '#94a3b8'}"/>
            <span class="text-foreground/80">layer {l}</span>
          </div>
        {/each}
        <div class="border-t border-border/40 pt-1.5 mt-1 space-y-0.5 pointer-events-auto">
          <button
            class="flex w-full items-center gap-1.5 text-left rounded px-1 py-0.5 transition-opacity hover:bg-white/5 cursor-pointer"
            class:opacity-35={!showAttention}
            on:click={() => showAttention = !showAttention}
          >
            <span class="inline-block w-2 h-2 rounded-full bg-amber-400 flex-shrink-0"/>
            <span class="text-[10px] font-mono">attention synapse</span>
            <span class="ml-auto text-[9px] opacity-60">{showAttention ? 'on' : 'off'}</span>
          </button>
          <button
            class="flex w-full items-center gap-1.5 text-left rounded px-1 py-0.5 transition-opacity hover:bg-white/5 cursor-pointer"
            class:opacity-35={!showResidual}
            on:click={() => showResidual = !showResidual}
          >
            <span class="inline-block w-2 h-2 rounded-full bg-sky-400 flex-shrink-0"/>
            <span class="text-[10px] font-mono">residual (cross-layer)</span>
            <span class="ml-auto text-[9px] opacity-60">{showResidual ? 'on' : 'off'}</span>
          </button>
        </div>
      </div>
    </div>
  {:else}
    <div class="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
      <div class="text-center font-mono text-xs text-muted-foreground">
        <div class="text-4xl mb-3 opacity-60">⬡</div>
        <div class="font-medium">BDH Layer Graph</div>
        <div class="text-[10px] mt-1.5 opacity-80">synapses form as training progresses</div>
      </div>
    </div>
  {/if}
</div>

<style>
  .studio-graph-root {
    min-height: 100%;
  }
  .studio-graph-glass-bg {
    position: absolute;
    inset: 0;
    z-index: 0;
    background: hsl(var(--surface) / 0.35);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 0.5rem;
  }
  .glass-badge {
    background: hsl(var(--surface) / 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
  .glass-badge-muted {
    background: hsl(var(--surface) / 0.5);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
</style>
