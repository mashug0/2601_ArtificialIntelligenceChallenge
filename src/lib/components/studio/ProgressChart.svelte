<script lang="ts">
  export let losses: number[] = [];
  export let valAccs: number[] = [];
  export let width: number = 280;
  export let height: number = 120;

  const pad = { top: 10, right: 10, bottom: 24, left: 36 };
  const W = width - pad.left - pad.right;
  const H = height - pad.top - pad.bottom;

  function scale(arr: number[], min: number, max: number, h: number) {
    const range = max - min || 1;
    return arr.map(v => h - ((v - min) / range) * h);
  }

  function toPath(ys: number[], n: number): string {
    if (!ys.length) return '';
    const step = W / Math.max(n - 1, 1);
    return ys.map((y, i) => `${i === 0 ? 'M' : 'L'} ${i * step} ${y}`).join(' ');
  }

  $: lossMin = Math.min(0, ...losses);
  $: lossMax = Math.max(0.01, ...losses);
  $: accMin  = Math.min(0, ...valAccs);
  $: accMax  = Math.max(1, ...valAccs);

  $: lossYs = scale(losses, lossMin, lossMax, H);
  $: accYs  = scale(valAccs, accMin, accMax, H);
  $: n = Math.max(losses.length, valAccs.length, 2);

  // Y-axis labels
  $: lossLabels = [lossMin, (lossMin + lossMax) / 2, lossMax].map(v => v.toFixed(2));
  $: accLabels  = [accMax, (accMin + accMax) / 2, accMin].map(v => v.toFixed(0) + '%');
</script>

<svg {width} {height} class="font-mono text-xs select-none">
  <g transform="translate({pad.left},{pad.top})">
    <!-- Grid -->
    {#each [0, 0.5, 1] as t}
      <line x1="0" y1={H * (1 - t)} x2={W} y2={H * (1 - t)} stroke="hsl(240 20% 22%)" stroke-dasharray="3,3"/>
    {/each}

    <!-- Loss line (neon amber) -->
    {#if losses.length > 1}
      <path d={toPath(lossYs, n)} fill="none" stroke="hsl(40 100% 50%)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    {/if}

    <!-- Val Acc line (neon emerald) -->
    {#if valAccs.length > 1}
      <path d={toPath(accYs, n)} fill="none" stroke="hsl(150 100% 50%)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="4,2"/>
    {/if}

    <!-- Axes -->
    <line x1="0" y1="0" x2="0" y2={H} stroke="hsl(240 20% 30%)" stroke-width="1"/>
    <line x1="0" y1={H} x2={W} y2={H} stroke="hsl(240 20% 30%)" stroke-width="1"/>

    <!-- X label -->
    <text x={W / 2} y={H + 18} text-anchor="middle" fill="hsl(215 20% 45%)" font-size="9">epochs</text>

    <!-- Y labels (loss left) -->
    {#each lossLabels as label, i}
      <text x="-4" y={i === 0 ? H : i === 1 ? H / 2 + 4 : 4} text-anchor="end" fill="hsl(40 100% 55%)" font-size="8">{label}</text>
    {/each}
  </g>

  <!-- Legend -->
  <g transform="translate({pad.left}, {height - 6})">
    <line x1="0" y1="0" x2="12" y2="0" stroke="hsl(40 100% 50%)" stroke-width="1.5"/>
    <text x="15" y="4" fill="hsl(40 100% 55%)" font-size="8">loss</text>
    <line x1="42" y1="0" x2="54" y2="0" stroke="hsl(150 100% 50%)" stroke-width="1.5" stroke-dasharray="4,2"/>
    <text x="57" y="4" fill="hsl(150 100% 55%)" font-size="8">val%</text>
  </g>
</svg>
